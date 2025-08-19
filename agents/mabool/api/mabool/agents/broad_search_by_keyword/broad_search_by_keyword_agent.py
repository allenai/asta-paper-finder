import logging

from ai2i.chain import LLMModel, Timeouts, define_llm_endpoint
from ai2i.config import config_value
from ai2i.dcollection import BASIC_FIELDS, DocumentCollectionSortDef, ExtractedYearlyTimeRange
from ai2i.di import DI
from pydantic import Field

from mabool.agents.broad_search_by_keyword.broad_search_by_keyword_prompts import (
    broad_search,
)
from mabool.agents.common.common import (
    AgentState,
)
from mabool.agents.common.computed_fields.relevance import relevance_judgement_field
from mabool.agents.common.domain_utils import get_fields_of_study_filter_from_domains
from mabool.agents.common.relevance_judgement_utils import get_relevant_docs
from mabool.agents.common.sorting import SortPreferences, add_weighted_sort_score
from mabool.agents.common.utils import alog_args
from mabool.agents.complex_search.definitions import BroadSearchInput
from mabool.data_model.agent import (
    AgentError,
    AgentOutput,
    DomainsIdentified,
    RelevanceCriteria,
)
from mabool.data_model.config import cfg_schema
from mabool.infra.operatives import (
    CompleteResponse,
    Operative,
    OperativeResponse,
    PartialResponse,
    VoidResponse,
)
from mabool.utils.dc import DC
from mabool.utils.llm_utils import get_api_key_for_model

logger = logging.getLogger(__name__)


async def suggest_retrieval_query(paper_description: str) -> str:
    llm_model = LLMModel.from_name(config_value(cfg_schema.broad_search_by_keyword_agent.formulation_model_name))
    endpoint = define_llm_endpoint(
        default_timeout=Timeouts.medium,
        default_model=llm_model,
        api_key=get_api_key_for_model(llm_model),
    )

    return await endpoint.execute(broad_search).once(paper_description)


BroadSearchByKeywordInput = BroadSearchInput


class BroadSearchByKeywordState(AgentState):
    search_iteration: int = Field(default=1)


BroadSearchByKeywordOutput = AgentOutput


class BroadSearchByKeywordAgent(
    Operative[BroadSearchByKeywordInput, BroadSearchByKeywordOutput, BroadSearchByKeywordState]
):
    def register(self) -> None: ...

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self, state: BroadSearchByKeywordState | None, inputs: BroadSearchByKeywordInput
    ) -> tuple[BroadSearchByKeywordState | None, OperativeResponse[BroadSearchByKeywordOutput]]:
        state = state or BroadSearchByKeywordState(checkpoint=DC.empty())
        response = await self.search(
            inputs.content_query,
            inputs.relevance_criteria,
            inputs.domains,
            inputs.recent_first,
            inputs.recent_last,
            inputs.central_first,
            inputs.central_last,
            inputs.apply_relevance_judgement,
            inputs.time_range,
            inputs.venues,
            inputs.authors,
            state.search_iteration,
        )

        match response:
            case VoidResponse():
                return None, VoidResponse(
                    error=AgentError(
                        type="other", message=f"BroadSearchByKeywordAgent failed to respond; {response.error.message}"
                    )
                )
            case PartialResponse():
                return BroadSearchByKeywordState(
                    checkpoint=response.data.doc_collection, search_iteration=state.search_iteration + 1
                ), PartialResponse(
                    data=BroadSearchByKeywordOutput(
                        doc_collection=response.data.doc_collection,
                        response_text=response.data.response_text,
                    ),
                    error=AgentError(
                        type="other",
                        message=(
                            f"BroadSearchByKeywordAgent failed to respond; {response.error.message}"
                            if response.error
                            else "BroadSearchByKeywordAgent failed to respond."
                        ),
                    ),
                )
            case CompleteResponse():
                return BroadSearchByKeywordState(
                    checkpoint=response.data.doc_collection, search_iteration=state.search_iteration + 1
                ), CompleteResponse(
                    data=BroadSearchByKeywordOutput(
                        doc_collection=response.data.doc_collection,
                        response_text=response.data.response_text,
                    )
                )

    @DI.managed
    async def search(
        self,
        content_query: str,
        relevance_criteria: RelevanceCriteria,
        domains: DomainsIdentified,
        recent_first: bool,
        recent_last: bool,
        central_first: bool,
        central_last: bool,
        apply_relevance_judgement: bool,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        authors: list[str] | None = None,
        search_iteration: int = 1,
    ) -> OperativeResponse[BroadSearchByKeywordOutput]:
        if not content_query:
            raise ValueError("You need to provide a content query.")

        # Suggest a ranked list of possible queries for the retrieval engine
        query = await suggest_retrieval_query(content_query)

        if recent_first or recent_last or central_first or central_last:
            # Take `results_limit` times some factor, as later we will take top `results_limit`
            # results after sorting by citation count and/or year

            limit = config_value(cfg_schema.broad_search_by_keyword_agent.results_limit) * config_value(
                cfg_schema.broad_search_by_keyword_agent.extra_results_factor
            )

            # Perform search using the retrieval engine
            search_results = await DC.from_s2_search(
                query=query,
                limit=limit,
                search_iteration=search_iteration,
                time_range=time_range,
                venues=venues,
                fields_of_study=get_fields_of_study_filter_from_domains(domains),
                fields=[*BASIC_FIELDS, "citation_count"],
            )

            sort_prefs = SortPreferences(
                recent_first=recent_first,
                recent_last=recent_last,
                central_first=central_first,
                central_last=central_last,
                relevance_criteria=relevance_criteria,
            )
            search_results = await add_weighted_sort_score(search_results, sort_prefs)

            search_results = search_results.sorted(
                [DocumentCollectionSortDef(field_name="weighted_sort_score", order="desc")]
            ).take(config_value(cfg_schema.broad_search_by_keyword_agent.results_limit))

            search_results = await DC.from_docs(search_results.documents).with_fields(BASIC_FIELDS)
        else:
            search_results = await DC.from_s2_search(
                query=query,
                limit=config_value(cfg_schema.broad_search_by_keyword_agent.results_limit),
                search_iteration=search_iteration,
                time_range=time_range,
                venues=venues,
                fields_of_study=get_fields_of_study_filter_from_domains(domains),
            )

        if apply_relevance_judgement:
            search_results = await search_results.with_fields([relevance_judgement_field(relevance_criteria)])
            search_results = get_relevant_docs(search_results)

        return CompleteResponse(data=BroadSearchByKeywordOutput(response_text="", doc_collection=search_results))
