import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncGenerator, TypedDict

from aiocache import cached
from aiocache.serializers import JsonSerializer
from fastapi import APIRouter
from fastapi.exceptions import HTTPException
from fastapi.responses import Response
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tracers.context import register_configure_hook

from mabool.agents.paper_finder.definitions import PaperFinderInput
from mabool.agents.paper_finder.paper_finder_agent import run_agent
from mabool.api.route_utils import create_json_response
from mabool.data_model.agent import AgentOperationMode
from mabool.data_model.ids import generate_conversation_thread_id
from mabool.data_model.rounds import RoundRequest
from mabool.infra.operatives import CompleteResponse, PartialResponse, VoidResponse
from mabool.services.prioritized_task import DEFAULT_PRIORITY, PrioritySemaphore
from mabool.utils.dc import DC
from mabool.utils.file_based_cache import FileBasedCache

logger = logging.getLogger(__name__)
round_semaphore = PrioritySemaphore(concurrency=3)

router = APIRouter(tags=["rounds"], prefix="/api/2/rounds")


@cached(cache=FileBasedCache, serializer=JsonSerializer())
async def run_round_with_cache(
    paper_description: str, anchor_corpus_ids: list[str], operation_mode: AgentOperationMode
) -> dict:
    conversation_thread_id = generate_conversation_thread_id()

    inp = PaperFinderInput(
        doc_collection=DC.empty(),
        query=paper_description,
        anchor_corpus_ids=anchor_corpus_ids,
        operation_mode=operation_mode or "infer",
    )

    # not catching asyncio.QueueFull as wev'e got nothing to do with it,
    # let's allow it to propagate
    async with round_semaphore.priority_context(DEFAULT_PRIORITY):
        async with get_mabool_callback() as cb:
            response = await run_agent(inp, conversation_thread_id)

    match response:
        case VoidResponse():
            raise HTTPException(status_code=500, detail=response.error.message)
        case CompleteResponse() | PartialResponse():
            result_set_to_return = response.data.model_dump(mode="json")
            result_set_to_return["token_breakdown_by_model"] = cb.tokens_by_model
            result_set_to_return["session_id"] = conversation_thread_id
            return result_set_to_return
        case _:
            raise HTTPException(status_code=500, detail="Unexpected response type from agent")


@router.post("")
async def start_round(round_request: RoundRequest) -> Response:
    logger.info(f"start_round called: {round_request=}")

    return create_json_response(
        await run_round_with_cache(
            round_request.paper_description,
            [],
            round_request.operation_mode or "infer",
            cache_read=round_request.read_results_from_cache,
        )
    )


class TokenUsage(TypedDict):
    total: int
    prompt: int
    completion: int
    reasoning: int


class MaboolCallbackHandler(AsyncCallbackHandler):
    tokens_by_model: dict[str, TokenUsage]

    def __init__(self) -> None:
        super().__init__()
        self.tokens_by_model = {}

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        generation = response.generations[0][0]
        model_name = "unknown"
        if isinstance(generation, ChatGeneration):
            try:
                message = generation.message
                if isinstance(message, AIMessage):
                    usage_metadata = message.usage_metadata
                    response_metadata = message.response_metadata
                    if "model_name" in response_metadata:
                        model_name = response_metadata["model_name"]
                    elif "model_version" in response_metadata:
                        model_name = response_metadata["model_version"]
                    else:
                        model_name = "unknown"
                else:
                    usage_metadata = None
            except AttributeError:
                usage_metadata = None
        else:
            usage_metadata = None

        if usage_metadata:
            if model_name not in self.tokens_by_model:
                self.tokens_by_model[model_name] = {
                    "total": 0,
                    "prompt": 0,
                    "completion": 0,
                    "reasoning": 0,
                }
            self.tokens_by_model[model_name]["total"] += usage_metadata["total_tokens"]
            self.tokens_by_model[model_name]["prompt"] += usage_metadata["input_tokens"]
            self.tokens_by_model[model_name]["completion"] += usage_metadata["output_tokens"]
            self.tokens_by_model[model_name]["reasoning"] += usage_metadata.get("output_token_details", {}).get(
                "reasoning", 0
            )


mabool_callback_var: ContextVar[MaboolCallbackHandler | None] = ContextVar("mabool_callback", default=None)
register_configure_hook(mabool_callback_var, True)


@asynccontextmanager
async def get_mabool_callback() -> AsyncGenerator[MaboolCallbackHandler, None]:
    cb = MaboolCallbackHandler()
    mabool_callback_var.set(cb)
    yield cb
    mabool_callback_var.set(None)
