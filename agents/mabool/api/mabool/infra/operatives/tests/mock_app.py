import logging
from asyncio import sleep
from typing import Any

from ai2i.config import application_config_ctx, load_conf
from ai2i.di import create_empty_app_context, create_managed_app
from fastapi import FastAPI, Request
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from mabool.data_model.agent import AgentError
from mabool.infra.operatives import (
    CompleteResponse,
    InquiryQuestion,
    InquiryReply,
    Operative,
    OperativeResponse,
    PartialResponse,
    VoidResponse,
    operative_session,
)
from mabool.utils.paths import project_root


class AnimalQueryAgentState(BaseModel):
    user_identity: str | None = None


class AnimalQuery(BaseModel):
    query: str


class LimbsQuery(BaseModel):
    animal: str


class TailQuery(BaseModel):
    animal: str


class TailAgent(Operative[TailQuery, bool, AnimalQueryAgentState]):
    async def handle_operation(
        self, state: AnimalQueryAgentState | None, inputs: TailQuery
    ) -> tuple[AnimalQueryAgentState | None, OperativeResponse[bool]]:
        if inputs.animal == "box":
            return state, CompleteResponse(data=True)
        elif inputs.animal == "bat":
            return state, CompleteResponse(data=False)
        else:
            inquiry = self.inquiry()
            if inquiry is None:
                return state, CompleteResponse(data=True)
            else:
                do_you_have_a_tail = await inquiry.ask(
                    InquiryQuestion(question="Do you have a tail?", options=[True, False])
                )
                return state, CompleteResponse(data=bool(do_you_have_a_tail.answer))


class LimbCounterTool(Operative[LimbsQuery, int, None]):
    async def handle_operation(self, state: None, inputs: LimbsQuery) -> tuple[None, OperativeResponse[int]]:
        if inputs.animal == "fox":
            return state, CompleteResponse(data=4)
        elif inputs.animal == "bat":
            return state, CompleteResponse(data=2)
        return state, VoidResponse(error=AgentError(type="other", message="Unknown animal."))


class InteractiveAnimalQueryAgent(Operative[AnimalQuery, list[str], AnimalQueryAgentState]):
    def register(self) -> None:
        self.tail_agent = self.init_operative("tail_agent", TailAgent)

    async def handle_operation(
        self, state: AnimalQueryAgentState | None, inputs: AnimalQuery
    ) -> tuple[AnimalQueryAgentState | None, OperativeResponse[list[str]]]:
        if inputs.query == "wait":
            await sleep(2)
            return state, CompleteResponse(data=["wait", "wait", "wait"])

        if inputs.query == "blah":
            return state, VoidResponse(error=AgentError(type="other", message="You have to ask about animals."))

        if inputs.query == "who are you?" and state and state.user_identity:
            return state, CompleteResponse(data=[state.user_identity])

        inquiry = self.inquiry()
        who_are_you_response = None
        if inquiry:
            who_are_you_response = await inquiry.ask(
                InquiryQuestion(question="Who are you?", options=["fox", "bat", "box"])
            )

            who_are_you_response = await inquiry.ask(
                InquiryQuestion(question="Who are you?", options=["fox", "bat", "box"])
            )
            logging.debug(who_are_you_response.answer)

            who_are_you_response = await inquiry.ask(
                InquiryQuestion(question="Who are you?", options=["fox", "bat", "box"])
            )
        else:
            who_are_you_response = InquiryReply(answer="fox")

        limb_counter = self.init_operative("limb_counter", LimbCounterTool)
        limbs = await limb_counter(LimbsQuery(animal=who_are_you_response.answer))
        logging.debug(limbs)

        tail_response = await self.tail_agent(TailQuery(animal=who_are_you_response.answer))
        logging.debug(tail_response)

        if inquiry:
            new_state = AnimalQueryAgentState(user_identity=who_are_you_response.answer)
            if who_are_you_response.answer == "fox":
                return new_state, CompleteResponse(data=["sight", "smell", "sound"])
            elif who_are_you_response.answer == "bat":
                return new_state, PartialResponse(
                    data=["echolocation"], error=AgentError(type="other", message="some methods are missing")
                )
            else:
                return state, VoidResponse(error=AgentError(type="other", message="Wrong option."))
        else:
            return state, VoidResponse(error=AgentError(type="other", message="No interactions available."))


def create_mock_app(**config_overrides: dict[str, Any]) -> FastAPI:
    conf_dir = project_root() / "conf"
    conf = load_conf(conf_dir)
    mock_app_context = create_empty_app_context()
    config_settings = conf.config.merge_dict(config_overrides)
    with application_config_ctx(conf):
        mock_app = create_managed_app(mock_app_context, config_settings, {}, conf_dir)

    mock_app.add_middleware(SessionMiddleware, secret_key="secret")

    @mock_app.post("/interactive_animals")
    async def interactive_animals(
        query: AnimalQuery | InquiryReply, req: Request
    ) -> OperativeResponse[list[str]] | InquiryQuestion:
        return await run_interactive_animals(query, req.headers.get("Conversation-Thread-Id"))  # type: ignore

    @mock_app.post("/non_interactive_animals")
    async def non_interactive_animals(
        query: AnimalQuery | InquiryReply, req: Request
    ) -> OperativeResponse[list[str]] | InquiryQuestion:
        return await run_non_interactive_animals(query, req.headers.get("Conversation-Thread-Id"))  # type: ignore

    return mock_app


@operative_session(InteractiveAnimalQueryAgent, "interactive_animals")  # type: ignore
async def run_interactive_animals(
    query: AnimalQuery | InquiryReply, conversation_thread_id: str
) -> OperativeResponse[list[str]] | InquiryQuestion:
    # Handled by the decorator.
    ...


@operative_session(InteractiveAnimalQueryAgent, "interactive_animals", allow_interactions=False)  # type: ignore
async def run_non_interactive_animals(
    query: AnimalQuery | InquiryReply, conversation_thread_id: str
) -> OperativeResponse[list[str]] | InquiryQuestion:
    # Handled by the decorator.
    ...
