import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Iterator, Protocol, Sequence

from ai2i.config import config_value
from ai2i.dcollection import DocumentCollection
from pydantic import ConfigDict, PrivateAttr

from mabool.agents.broad_search_by_keyword.broad_search_by_keyword_agent import (
    BroadSearchByKeywordAgent,
)
from mabool.agents.by_citing_papers.broad_by_specific_paper_citation_agent import (
    BroadBySpecificPaperCitationAgent,
)
from mabool.agents.common.common import AgentState, filter_docs_by_metadata
from mabool.agents.common.computed_fields.relevance import relevance_judgement_field
from mabool.agents.common.relevance_judgement_utils import (
    get_relevant_docs,
    log_relevance_value_counts,
    report_relevance_judgement_counts,
)
from mabool.agents.common.utils import alog_args
from mabool.agents.complex_search.definitions import (
    BroadSearchArgs,
    BroadSearchInput,
    BroadSearchOutput,
)
from mabool.agents.dense.dense_agent import DenseAgent
from mabool.agents.dense.relevance_loading_optimization import (
    HighlyRelevantShortcircuit,
    Shortcircuit,
    adaptive_load,
    post_relevance_judgement_loading,
)
from mabool.agents.llm_suggestion.llm_suggestion_agent import LLMSuggestionAgent
from mabool.agents.snowball.snowball_agent import SnowballAgent
from mabool.data_model.agent import AgentError
from mabool.data_model.config import cfg_schema
from mabool.infra.operatives import (
    CompleteResponse,
    Operative,
    OperativeResponse,
    PartialResponse,
    VoidResponse,
)
from mabool.utils.asyncio import custom_gather
from mabool.utils.dc import DC

logger = logging.getLogger(__name__)

BroadSearchState = AgentState


class SearchOperative[INPUT, OUTPUT](Protocol):
    async def __call__(self, inputs: INPUT) -> OperativeResponse[OUTPUT]: ...


class ShortcircuitOperative:
    def __init__(
        self,
        operative: SearchOperative,
        shortcircuit: Shortcircuit,
    ):
        self._operative = operative
        self._shortcircuit = shortcircuit

    async def __call__(self, inputs: BroadSearchInput) -> OperativeResponse[BroadSearchOutput]:
        if self._shortcircuit.should_break():
            logger.info("Shortcircuit condition met, skipping operation")
            return CompleteResponse(
                data=BroadSearchOutput(
                    doc_collection=DC.empty(),
                )
            )

        return await self._operative(inputs)


@dataclass(frozen=True)
class SearchIterationSources:
    primary_sources: list[SearchOperative]
    followup_sources: list[SearchOperative]


class BroadSearchCommand(BroadSearchArgs):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dense_agent: DenseAgent
    snowball_agent: SnowballAgent
    by_keyword_agent: BroadSearchByKeywordAgent
    specific_paper_citation_agent: BroadBySpecificPaperCitationAgent
    llm_suggestion_agent: LLMSuggestionAgent
    _search_iterations_doc_sources: list[SearchIterationSources] = PrivateAttr()
    _relevance_cap_shortcircuit: HighlyRelevantShortcircuit = PrivateAttr()
    _relevance_cap_iteration_shortcircuit: HighlyRelevantShortcircuit = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._relevance_cap_shortcircuit = HighlyRelevantShortcircuit(
            score_cap=config_value(cfg_schema.relevance_judgement.highly_relevant_cap)
        )
        self._relevance_cap_iteration_shortcircuit = HighlyRelevantShortcircuit(
            score_cap=config_value(cfg_schema.relevance_judgement.highly_relevant_cap)
        )
        self._search_iterations_doc_sources = self._collect_doc_sources(
            self._relevance_cap_shortcircuit,
            self.dense_agent,
            self.snowball_agent,
            self.by_keyword_agent,
            self.specific_paper_citation_agent,
            self.llm_suggestion_agent,
        )

    async def execute(self) -> OperativeResponse[BroadSearchOutput]:
        self._validate_args()

        docs = DC.empty()
        with break_iteration_context():
            for i, doc_sources in enumerate(self._search_iterations_doc_sources):
                logger.info(f"=== Iteration {i + 1} ===")
                self._refresh_iteration_shortcircuit()
                docs = await self._fetch_from_doc_sources(doc_sources, docs)
                docs = await self._apply_relevance_judgement(docs)
        docs = get_relevant_docs(docs)
        docs = await filter_docs_by_metadata(docs, self.time_range, self.venues)
        return CompleteResponse(
            data=BroadSearchOutput(
                doc_collection=docs,
            )
        )

    def _collect_doc_sources(
        self,
        shortcircuit: Shortcircuit,
        dense_agent: DenseAgent,
        snowball_agent: SnowballAgent,
        by_keyword_agent: BroadSearchByKeywordAgent,
        specific_paper_citation_agent: BroadBySpecificPaperCitationAgent,
        llm_suggestion_agent: LLMSuggestionAgent,
    ) -> list[SearchIterationSources]:
        suitable_for_by_citing = self.suitable_for_by_citing or len(self.anchor_doc_collection) > 0
        sc_dense_agent = ShortcircuitOperative(operative=dense_agent, shortcircuit=shortcircuit)
        if suitable_for_by_citing:
            return [
                SearchIterationSources(
                    primary_sources=[
                        sc_dense_agent,
                        by_keyword_agent,
                        specific_paper_citation_agent,
                        llm_suggestion_agent,
                    ],
                    followup_sources=[snowball_agent],
                ),
                SearchIterationSources(
                    primary_sources=[sc_dense_agent],
                    followup_sources=[snowball_agent],
                ),
            ]
        else:
            return [
                SearchIterationSources(
                    primary_sources=[sc_dense_agent, by_keyword_agent, llm_suggestion_agent],
                    followup_sources=[snowball_agent],
                ),
                SearchIterationSources(
                    primary_sources=[sc_dense_agent],
                    followup_sources=[snowball_agent],
                ),
            ]

    def _validate_args(self) -> None:
        if not self.content_query:
            raise ValueError("You need to provide a content query.")
        if self.recent_first and self.recent_last:
            raise ValueError("You can't have both recent_first and recent_last constraints.")
        if self.central_first and self.central_last:
            raise ValueError("You can't have both central_first and central_last constraints.")

    def _refresh_iteration_shortcircuit(self) -> None:
        self._relevance_cap_iteration_shortcircuit = HighlyRelevantShortcircuit(
            score_cap=config_value(cfg_schema.relevance_judgement.highly_relevant_cap)
        )

    async def _fetch_from_operatives(
        self, operatives: list[SearchOperative], docs: DocumentCollection
    ) -> Sequence[OperativeResponse]:
        doc_source_responses: Sequence[OperativeResponse] = await custom_gather(
            *(
                doc_source(
                    BroadSearchInput(
                        user_input=self.user_input,
                        content_query=self.content_query,
                        relevance_criteria=self.relevance_criteria,
                        extracted_name=self.extracted_name,
                        recent_first=self.recent_first,
                        recent_last=self.recent_last,
                        central_first=self.central_first,
                        central_last=self.central_last,
                        time_range=self.time_range,
                        venues=self.venues,
                        authors=self.authors,
                        domains=self.domains,
                        anchor_doc_collection=self.anchor_doc_collection,
                        doc_collection=docs,
                        apply_relevance_judgement=False,
                        n_suggestions=config_value(cfg_schema.broad_search_agent.llm_n_suggestions),
                    )
                )
                for doc_source in operatives
            )
        )
        return doc_source_responses

    async def _fetch_from_doc_sources(
        self, doc_sources: SearchIterationSources, docs: DocumentCollection
    ) -> DocumentCollection:
        primary_sources_responses = await self._fetch_from_operatives(doc_sources.primary_sources, docs)
        response_docs = await self._merge_doc_source_responses(docs, primary_sources_responses)

        if doc_sources.followup_sources:
            followup_doc_source_responses = await self._fetch_from_operatives(
                doc_sources.followup_sources, response_docs
            )

            response_docs += await self._merge_doc_source_responses(response_docs, followup_doc_source_responses)

        return response_docs

    async def _merge_doc_source_responses(
        self, docs: DocumentCollection, doc_source_responses: Sequence[OperativeResponse]
    ) -> DocumentCollection:
        void_responses, doc_collections = (
            [r for r in doc_source_responses if isinstance(r, VoidResponse)],
            [r.data.doc_collection for r in doc_source_responses if not isinstance(r, VoidResponse)],
        )
        for void_response in void_responses:
            logger.exception(f"An error occurred while running a sub-agent: {void_response.error.message}")
        return DC.merge([docs, *doc_collections])

    async def _apply_relevance_judgement(self, docs: DocumentCollection) -> DocumentCollection:
        docs = docs.update_computed_fields([relevance_judgement_field(self.relevance_criteria)])
        relevance_judgement_quota = config_value(cfg_schema.relevance_judgement.quota)
        logger.info(f"Running relevance judgement, with {relevance_judgement_quota=}")
        docs, policy = await adaptive_load(
            documents=docs,
            field_name="relevance_judgement",
            docs_quota=relevance_judgement_quota,
            shortcircuits=[self._relevance_cap_iteration_shortcircuit, self._relevance_cap_shortcircuit],
        )
        await post_relevance_judgement_loading(
            docs, strategy=str(policy), with_optimal=config_value(cfg_schema.relevance_judgement.optimal_solution)
        )
        await log_relevance_value_counts(docs)
        await report_relevance_judgement_counts(docs)
        return docs


class BroadSearchAgent(Operative[BroadSearchInput, BroadSearchOutput, BroadSearchState]):
    """
    BroadSearchAgent is a sub-agent that handles queries that contain content,
    recency and centrality constraints.
    """

    def register(self) -> None:
        self.dense_agent = self.init_operative("dense_agent", DenseAgent)
        self.snowball_agent = self.init_operative("snowball_agent", SnowballAgent)
        self.broad_search_by_keyword_agent = self.init_operative(
            "broad_search_by_keyword_agent", BroadSearchByKeywordAgent
        )
        self.specific_paper_citation_agent = self.init_operative(
            "specific_paper_citation_agent", BroadBySpecificPaperCitationAgent
        )
        self.llm_suggestion_agent = self.init_operative("llm_suggestion_agent", LLMSuggestionAgent)

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self, state: BroadSearchState | None, inputs: BroadSearchInput
    ) -> tuple[BroadSearchState | None, OperativeResponse[BroadSearchOutput]]:
        search_command = BroadSearchCommand(
            dense_agent=self.dense_agent,
            snowball_agent=self.snowball_agent,
            by_keyword_agent=self.broad_search_by_keyword_agent,
            specific_paper_citation_agent=self.specific_paper_citation_agent,
            llm_suggestion_agent=self.llm_suggestion_agent,
            recent_first=inputs.recent_first,
            recent_last=inputs.recent_last,
            central_first=inputs.central_first,
            central_last=inputs.central_last,
            suitable_for_by_citing=inputs.suitable_for_by_citing,
            user_input=inputs.user_input,
            content_query=inputs.content_query,
            relevance_criteria=inputs.relevance_criteria,
            extracted_name=inputs.extracted_name,
            time_range=inputs.time_range,
            venues=inputs.venues,
            authors=inputs.authors,
            domains=inputs.domains,
            anchor_doc_collection=inputs.anchor_doc_collection,
        )
        response = await search_command.execute()
        return self._handle_response(response)

    @staticmethod
    def _handle_response(
        response: OperativeResponse[BroadSearchOutput],
    ) -> tuple[BroadSearchState | None, OperativeResponse[BroadSearchOutput]]:
        match response:
            case VoidResponse():
                return None, VoidResponse(
                    error=AgentError(
                        type="other",
                        message=f"BroadSearchAgent failed to respond; {response.error.message}",
                    )
                )
            case PartialResponse():
                return BroadSearchState(checkpoint=response.data.doc_collection), PartialResponse(
                    data=BroadSearchOutput(
                        doc_collection=response.data.doc_collection,
                    ),
                    error=AgentError(
                        type="other",
                        message=(
                            f"BroadSearchAgent failed to respond; {response.error.message}"
                            if response.error
                            else "BroadSearchAgent failed to respond."
                        ),
                    ),
                )
            case CompleteResponse():
                return BroadSearchState(checkpoint=response.data.doc_collection), CompleteResponse(
                    data=BroadSearchOutput(
                        doc_collection=response.data.doc_collection,
                    )
                )
            case _:
                raise ValueError(f"Unknown response type: {response.__class__.__name__}")


@contextlib.contextmanager
def break_iteration_context() -> Iterator:
    try:
        yield
    except StopIteration:
        pass
