import logging
from typing import Sequence

from ai2i.chain import LLMEndpoint, LLMModel, Timeouts, define_llm_endpoint
from ai2i.config import config_value
from ai2i.dcollection import DocumentCollection, ExtractedYearlyTimeRange, OriginQuery
from ai2i.di import DI
from pydantic import BaseModel

from mabool.agents.common.common import (
    AgentState,
)
from mabool.agents.common.domain_utils import get_system_domain_params
from mabool.agents.common.utils import alog_args
from mabool.agents.llm_suggestion.llm_suggestion_prompts import (
    SuggestedPaper,
    SuggestPapersInput,
    suggested_paper,
)
from mabool.data_model.agent import (
    AgentInput,
    AgentOutput,
    DomainsIdentified,
)
from mabool.data_model.config import cfg_schema
from mabool.infra.operatives import CompleteResponse, Operative, OperativeResponse
from mabool.utils.asyncio import custom_gather
from mabool.utils.dc import DC
from mabool.utils.llm_utils import get_api_key_for_model

logger = logging.getLogger(__name__)


class LLMSuggestionArgs(BaseModel):
    user_input: str
    domains: DomainsIdentified
    extra_hints: str | None = None
    n_suggestions: int | None = 1


class LLMSuggestionInput(LLMSuggestionArgs, AgentInput):
    pass


LLMSuggestionState = AgentState
LLMSuggestionOutput = AgentOutput


def get_default_endpoint() -> LLMEndpoint:
    llm_model = LLMModel.from_name(config_value(cfg_schema.llm_suggestion_agent.llm_model_name))
    return define_llm_endpoint(
        default_timeout=Timeouts.medium,
        default_model=llm_model,
        logger=logger,
        api_key=get_api_key_for_model(llm_model),
    )


async def fetch_from_s2(
    suggested_papers: Sequence[SuggestedPaper],
) -> DocumentCollection:
    search_result_futures = [
        # reporting handled elsewhere in this file.
        DC.from_s2_by_title(
            suggested_paper.title,
            time_range=(
                ExtractedYearlyTimeRange(start=suggested_paper.year - 2, end=suggested_paper.year + 2)
                if suggested_paper.year
                else None
            ),
        )
        for suggested_paper in suggested_papers
    ]

    search_results = await custom_gather(*search_result_futures)

    if search_results:
        merged_search_results = DC.merge(search_results)
    else:
        merged_search_results = DC.from_docs([])

    return merged_search_results


@DI.managed
async def get_llm_suggested_papers(
    user_input: str,
    domains: DomainsIdentified,
    extra_hints: str | None = None,
    n_suggestions: int | None = None,
    search_iteration: int = 1,
) -> DocumentCollection:
    suggested_papers = (
        await get_default_endpoint()
        .execute(suggested_paper)
        .once(
            SuggestPapersInput(
                query=user_input,
                extra_hints=extra_hints or "",
                n_suggestions=n_suggestions or 1,
                **get_system_domain_params(domains),
            )
        )
    )
    if len(suggested_papers) == 0:
        return DC.empty()

    logger.info(f"Suggested papers: {suggested_papers}")
    found_papers = await fetch_from_s2(suggested_papers)

    # append "llm" to the origins list
    found_papers = found_papers.map_enumerate(
        lambda i, doc: doc.clone_with(
            {
                "origins": [
                    OriginQuery(
                        query_type="llm",
                        provider="openai",
                        variant=get_default_endpoint().default_model.name,
                        query=f"{user_input} | Extra hints: {extra_hints}",
                        iteration=search_iteration,
                        ranks=[i + 1],
                    )
                ]
            },
        )
    )

    return found_papers


class LLMSuggestionAgent(Operative[LLMSuggestionInput, LLMSuggestionOutput, LLMSuggestionState]):
    def register(self) -> None: ...

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self, state: LLMSuggestionState | None, inputs: LLMSuggestionInput
    ) -> tuple[LLMSuggestionState | None, OperativeResponse[LLMSuggestionOutput]]:
        search_results = await get_llm_suggested_papers(
            inputs.user_input,
            inputs.domains,
            inputs.extra_hints,
            inputs.n_suggestions,
        )

        return (
            state,
            CompleteResponse(data=LLMSuggestionOutput(response_text="", doc_collection=search_results)),
        )
