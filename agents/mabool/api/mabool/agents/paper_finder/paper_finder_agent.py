from __future__ import annotations

import logging
from typing import cast

from ai2i.config import config_value, ufv
from ai2i.dcollection import BASIC_FIELDS, UI_REQUIRED_FIELDS, DocumentCollection
from pydantic import BaseModel, ConfigDict

from mabool.agents.broad_search_by_keyword.broad_search_by_keyword_agent import (
    BroadSearchByKeywordAgent,
    BroadSearchByKeywordInput,
    BroadSearchByKeywordOutput,
)
from mabool.agents.common.common import (
    AgentState,
)
from mabool.agents.common.computed_fields.fields import final_agent_score_field
from mabool.agents.common.explain import (
    explain_query_analysis,
    generate_response_summary,
)
from mabool.agents.common.sorting import SortPreferences, sorted_docs_by_preferences
from mabool.agents.common.utils import alog_args
from mabool.agents.complex_search.broad_search import BroadSearchAgent
from mabool.agents.complex_search.definitions import BroadSearchInput, BroadSearchOutput
from mabool.agents.complex_search.fast_broad_search import FastBroadSearchAgent
from mabool.agents.llm_suggestion.llm_suggestion_agent import get_llm_suggested_papers
from mabool.agents.metadata_only.metadata_only_agent import (
    MetadataOnlySearchAgent,
    MetadataOnlySearchInput,
)
from mabool.agents.metadata_only.metadata_planner_agent import MetadataPlannerAgent, MetadataPlannerInput
from mabool.agents.paper_finder.anchor_docs import enrich_anchor_documents
from mabool.agents.paper_finder.definitions import PaperFinderInput
from mabool.agents.query_analyzer.anchor import combine_content_query_with_anchors
from mabool.agents.query_analyzer.query_analyzer import decompose_and_analyze_query_restricted
from mabool.agents.query_refusal.utils import clarify_query_refusal, refusal_requires_clarification, refusal_responses
from mabool.agents.search_by_authors.search_by_authors_agent import (
    SearchByAuthorsAgent,
    SearchByAuthorsInput,
    SearchByAuthorsOutput,
)
from mabool.agents.specific_paper_by_name.specific_paper_by_name_agent import (
    SpecificPaperByNameAgent,
    SpecificPaperByNameInput,
    SpecificPaperByNameOutput,
)
from mabool.agents.specific_paper_by_title.specific_paper_by_title_agent import (
    SpecificPaperByTitleAgent,
    SpecificPaperByTitleInput,
    SpecificPaperByTitleOutput,
)
from mabool.data_model.agent import (
    AgentError,
    AgentOutput,
    AnalyzedQuery,
    DomainsIdentified,
    ExplainedAgentOutput,
    PartiallyAnalyzedQuery,
    QueryAnalysisFailure,
    QueryAnalysisPartialSuccess,
    QueryAnalysisRefusal,
    QueryAnalysisSuccess,
    QueryAnalyzerError,
    QueryType,
)
from mabool.data_model.config import cfg_schema
from mabool.data_model.ids import ConversationThreadId
from mabool.data_model.specifications import PaperSpec, Specifications
from mabool.data_model.ufs import uf
from mabool.infra.operatives import (
    CompleteResponse,
    Operative,
    OperativeResponse,
    PartialResponse,
    VoidResponse,
    operative_session,
)
from mabool.infra.operatives.operatives import InquiryQuestion, InquiryReply
from mabool.utils.dc import DC
from mabool.utils.text import HRULE, join_paragraphs

logger = logging.getLogger(__name__)

PaperFinderState = AgentState


PaperFinderOutput = ExplainedAgentOutput


async def _add_missing_fields_and_sort(
    initial_docs: DocumentCollection, plan_context: PlanContext
) -> tuple[DocumentCollection, str]:
    docs = await initial_docs.with_fields(BASIC_FIELDS + UI_REQUIRED_FIELDS)
    return await plan_context.sort(docs)


class PaperFinderAgent(Operative[PaperFinderInput, PaperFinderOutput, PaperFinderState]):
    def register(self) -> None:
        self.broad_search_agent = self.init_operative("broad_search_agent", BroadSearchAgent)
        self.fast_broad_search_agent = self.init_operative("fast_broad_search_agent", FastBroadSearchAgent)
        self.broad_search_by_keyword = self.init_operative("broad_search_by_keyword_agent", BroadSearchByKeywordAgent)
        self.specific_paper_by_title = self.init_operative("specific_paper_by_title_agent", SpecificPaperByTitleAgent)
        self.search_by_authors = self.init_operative("search_by_authors_agent", SearchByAuthorsAgent)
        self.specific_paper_by_name = self.init_operative("specific_paper_by_name_agent", SpecificPaperByNameAgent)
        self.metadata_only_agent = self.init_operative("metadata_only_agent", MetadataOnlySearchAgent)
        self.metadata_planner_agent = self.init_operative("metadata_planner_agent", MetadataPlannerAgent)

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self, state: PaperFinderState | None, inputs: PaperFinderInput
    ) -> tuple[PaperFinderState | None, OperativeResponse[PaperFinderOutput]]:
        is_fast = inputs.operation_mode == "fast"
        query_analysis_result = await decompose_and_analyze_query_restricted(inputs.query)
        match query_analysis_result:
            case QueryAnalysisSuccess(
                analyzed_query=analyzed_query, specifications=Specifications() as specifications
            ) if specifications.is_non_trivial_metadata_only():
                logger.info("Running metadata planner on specifications")
                docs, aggregated_response_text = await self.run_metadata_planner_on_specifications(
                    inputs, specifications
                )
                if len(docs) == 0:
                    match specifications:
                        case Specifications(union=[PaperSpec(name=str())]):
                            anchor_docs = await enrich_anchor_documents(analyzed_query, inputs)
                            docs, aggregated_response_text = await self.retry_specific_in_broad_mode(
                                inputs, analyzed_query, anchor_docs
                            )
                        case _:
                            pass
                metadata_analyzed_query = analyzed_query.model_copy(update={"content": ""})
                planner_output = await get_paper_finder_output_from_docs(
                    docs, aggregated_response_text, inputs.query, metadata_analyzed_query, is_fast
                )
                return PaperFinderState(checkpoint=docs), CompleteResponse(data=planner_output)
            case QueryAnalysisSuccess(analyzed_query=analyzed_query):
                anchor_docs = await enrich_anchor_documents(analyzed_query, inputs)
                analyzed_query.content = await combine_content_query_with_anchors(analyzed_query.content, anchor_docs)
                docs, aggregated_response_text = await self.run_paper_finder_on_analyzed_query(
                    inputs, analyzed_query, anchor_docs
                )
                if len(docs) == 0 and analyzed_query.query_type.type in ["SPECIFIC_BY_TITLE", "SPECIFIC_BY_NAME"]:
                    # re-run in broad mode
                    docs, aggregated_response_text = await self.retry_specific_in_broad_mode(
                        inputs, analyzed_query, anchor_docs
                    )
                planner_output = await get_paper_finder_output_from_docs(
                    docs, aggregated_response_text, inputs.query, analyzed_query, is_fast
                )
                return PaperFinderState(checkpoint=docs), CompleteResponse(data=planner_output)
            case QueryAnalysisRefusal(analysis=analysis, errors=errors):
                # if we got here we have a refusal and a valid type
                assert analysis.possible_refusal.type is not None

                if refusal_requires_clarification[analysis.possible_refusal.type]:
                    anchor_docs = await enrich_anchor_documents(analysis, inputs)
                    (
                        docs,
                        aggregated_response_text,
                        choice_explanation,
                        choice,
                    ) = await self.run_paper_finder_on_refused_analyzed_query(inputs, analysis, anchor_docs, errors)
                    response_text_override = None
                    if choice == "cancel":
                        response_text_override = join_paragraphs(
                            ufv(uf.response_texts.paper_finder_agent.anything_else)
                        )
                else:
                    # this means soft rejection with a specified response text
                    response_text_override = refusal_responses()[analysis.possible_refusal.type]
                    docs = DC.empty()
                    choice_explanation = ufv(
                        uf.response_texts.paper_finder_agent.soft_rejection, refusal_type=analysis.possible_refusal.type
                    )
                    aggregated_response_text = ""

                return PaperFinderState(checkpoint=docs), PartialResponse(
                    data=await get_paper_finder_output_from_docs(
                        docs,
                        aggregated_response_text,
                        inputs.query,
                        analysis,
                        is_fast,
                        response_text_override,
                    ),
                    error=AgentError(type="query_refusal", message=choice_explanation),
                )
            case QueryAnalysisPartialSuccess(partially_analyzed_query=partially_analyzed_query, errors=errors):
                anchor_docs = await enrich_anchor_documents(partially_analyzed_query, inputs)
                docs, aggregated_response_text = await self.run_paper_finder_on_partially_analyzed_query(
                    inputs, partially_analyzed_query, errors
                )
                return PaperFinderState(checkpoint=docs), PartialResponse(
                    data=await get_paper_finder_output_from_docs(
                        docs, aggregated_response_text, inputs.query, partially_analyzed_query, is_fast
                    ),
                    error=AgentError(
                        type="other",
                        message=(
                            f"Got error(s) {', '.join([e.__class__.__name__ for e in errors])}; "
                            "fell back to LLM suggestions"
                        ),
                    ),
                )
            case QueryAnalysisFailure(partially_analyzed_query=partially_analyzed_query, error=error):
                return None, PartialResponse(
                    data=await get_paper_finder_output_from_docs(
                        docs=DC.empty(),
                        aggregated_response_text="",
                        user_query=inputs.query,
                        analyzed_input=partially_analyzed_query,
                        is_fast=is_fast,
                        response_text_override=error.message,
                    ),
                    error=AgentError(
                        type="no_actionable_data_query",
                        message=error.message,
                    ),
                )

    async def retry_specific_in_broad_mode(
        self, inputs: PaperFinderInput, analyzed_query: AnalyzedQuery, anchor_docs: DocumentCollection
    ) -> tuple[DocumentCollection, str]:
        retry_analyzed_query = analyzed_query.model_copy()
        retry_analyzed_query.query_type = QueryType(type="BROAD_BY_DESCRIPTION", broad_or_specific="broad")
        retry_inputs = inputs.model_copy()
        return await self.run_paper_finder_on_analyzed_query(retry_inputs, retry_analyzed_query, anchor_docs)

    async def run_metadata_planner_on_specifications(
        self, inputs: PaperFinderInput, specifications: Specifications
    ) -> tuple[DocumentCollection, str]:
        response = await self.metadata_planner_agent(
            MetadataPlannerInput(
                doc_collection=inputs.doc_collection,
                specification=specifications,
            )
        )
        match response:
            case CompleteResponse(data=AgentOutput(doc_collection=docs, response_text=aggregated_response_text)):
                pass
            case PartialResponse() | VoidResponse():
                raise Exception("MetadataPlannerAgent failed to respond;")
        return docs, aggregated_response_text

    async def run_paper_finder_on_analyzed_query(
        self, inputs: PaperFinderInput, analyzed_input: AnalyzedQuery, anchor_docs: DocumentCollection
    ) -> tuple[DocumentCollection, str]:
        response: OperativeResponse[AgentOutput] | OperativeResponse[BroadSearchOutput] | None = None
        plan_context = PlanContext(
            assume_recent_and_central_first=config_value(cfg_schema.assume_recent_and_central_first),
            should_sort=config_value(cfg_schema.should_sort),
            analyzed_input=analyzed_input,
            authors=analyzed_input.authors,
            venues=analyzed_input.venues,
            domains=analyzed_input.domains,
            anchor_doc_collection=anchor_docs,
            input_doc_collection=inputs.doc_collection,
            paper_finder_agent=self,
        )

        match plan_context.analyzed_input.query_type.type:
            case "SPECIFIC_BY_TITLE":
                response = await plan_context.run_specific_paper_by_title()
            case "SPECIFIC_BY_NAME":
                response = await plan_context.run_specific_paper_by_name()
            case "BY_AUTHOR":
                response = await plan_context.run_search_by_authors()
            case "BROAD_BY_DESCRIPTION":
                if inputs.operation_mode == "diligent":
                    response = await plan_context.run_broad_search()
                else:
                    response = await plan_context.run_fast_broad_search()
            case "METADATA_ONLY_NO_AUTHOR":
                response = await plan_context.attempt_metadata_only_search()
            case "UNKNOWN":
                raise Exception("PaperFinderAgent: Unknown query type in analyzed input")

        if isinstance(response, VoidResponse):
            raise Exception("A content Agent has failed to respond; " + response.error.message)

        docs, sorting_explanation = await _add_missing_fields_and_sort(response.data.doc_collection, plan_context)

        return docs, join_paragraphs(
            getattr(response.data, "response_text", ""),
            sorting_explanation,
        )

    async def run_paper_finder_on_refused_analyzed_query(
        self,
        inputs: PaperFinderInput,
        analyzed_input: AnalyzedQuery | PartiallyAnalyzedQuery,
        anchor_docs: DocumentCollection,
        errors: list[QueryAnalyzerError],
    ) -> tuple[DocumentCollection, str, str, str]:
        logger.info(
            "It seems the query would not benefit from a paper search, or not fully supported, clarifying further action"
        )
        choice = await clarify_query_refusal(self.inquiry(), analyzed_input)
        choice_explanation = ufv(uf.response_texts.paper_finder_agent.choice_explanation_prefix)
        match choice:
            case "run_anyway":
                if isinstance(analyzed_input, AnalyzedQuery):
                    logger.info("Running search normally...")
                    docs, aggregated_response_text = await self.run_paper_finder_on_analyzed_query(
                        inputs, analyzed_input, anchor_docs
                    )
                else:
                    docs, aggregated_response_text = await self.run_paper_finder_on_partially_analyzed_query(
                        inputs, cast(PartiallyAnalyzedQuery, analyzed_input), errors
                    )
                return (
                    docs,
                    aggregated_response_text,
                    ufv(uf.response_texts.paper_finder_agent.run_anyway, choice_explanation=choice_explanation),
                    choice,
                )
            case "cancel":
                return (
                    DC.empty(),
                    "",
                    ufv(uf.response_texts.paper_finder_agent.cancel, choice_explanation=choice_explanation),
                    choice,
                )

    async def run_paper_finder_on_partially_analyzed_query(
        self, inputs: PaperFinderInput, analyzed_input: PartiallyAnalyzedQuery, errors: list[QueryAnalyzerError]
    ) -> tuple[DocumentCollection, str]:
        logger.warning(
            f"Got error(s) in analyzed query {', '.join([e.__class__.__name__ for e in errors])}; falling back to LLM suggestions"
        )
        docs = await get_llm_suggested_papers(
            analyzed_input.original_query,
            domains=analyzed_input.domains
            if analyzed_input.domains
            else DomainsIdentified(main_field="Unknown", key_secondary_fields=[]),
            n_suggestions=config_value(cfg_schema.llm_suggestion_agent.fallback_n_suggestions),
        )
        return docs, ""


class PlanContext(BaseModel):
    assume_recent_and_central_first: bool
    should_sort: bool
    analyzed_input: AnalyzedQuery
    authors: list[str]
    venues: list[str]
    domains: DomainsIdentified
    anchor_doc_collection: DocumentCollection
    input_doc_collection: DocumentCollection
    paper_finder_agent: PaperFinderAgent
    consider_content_relevance: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def run_specific_paper_by_title(self) -> OperativeResponse[SpecificPaperByTitleOutput]:
        if not (
            self.analyzed_input.matched_title.matched_title and self.analyzed_input.matched_title.matched_corpus_ids
        ):
            raise AssertionError("Missing matched_title or matched_corpus_ids in by_title agent (shouldn't reach here)")
        self.should_sort = config_value(cfg_schema.specific_paper_by_title_agent.should_sort)
        by_title_response = await self.paper_finder_agent.specific_paper_by_title(
            SpecificPaperByTitleInput(
                matched_title=self.analyzed_input.matched_title.matched_title,
                matched_corpus_ids=self.analyzed_input.matched_title.matched_corpus_ids,
                time_range=self.analyzed_input.time_range,
                venues=self.venues,
                doc_collection=self.input_doc_collection,
            )
        )

        return by_title_response

    async def run_specific_paper_by_name(self) -> OperativeResponse[SpecificPaperByNameOutput]:
        self.should_sort = config_value(cfg_schema.specific_paper_by_name_agent.should_sort)
        if not self.analyzed_input.extracted_properties.specific_paper_name:
            return VoidResponse(
                error=AgentError(
                    type="other", message="SpecificPaperByNameAgent: cannot run without a specific paper name."
                )
            )
        return await self.paper_finder_agent.specific_paper_by_name(
            SpecificPaperByNameInput(
                user_input=self.analyzed_input.original_query,
                extracted_name=self.analyzed_input.extracted_properties.specific_paper_name,
                extracted_content=self.analyzed_input.content,
                time_range=self.analyzed_input.time_range,
                venues=self.venues,
                authors=self.authors,
                doc_collection=self.input_doc_collection,
                domains=self.domains,
            )
        )

    async def run_search_by_authors(self) -> OperativeResponse[SearchByAuthorsOutput]:
        if not self.analyzed_input.content:
            self.consider_content_relevance = False
        return await self.paper_finder_agent.search_by_authors(
            SearchByAuthorsInput(
                authors=self.authors,
                broad_or_specific=self.analyzed_input.query_type.broad_or_specific,
                user_content_input=self.analyzed_input.content,
                relevance_criteria=self.analyzed_input.relevance_criteria,
                time_range=self.analyzed_input.time_range,
                venues=self.venues,
                doc_collection=self.input_doc_collection,
                domains=self.domains,
            )
        )

    async def run_broad_search(self) -> OperativeResponse[BroadSearchOutput]:
        return await self.paper_finder_agent.broad_search_agent(
            BroadSearchInput(
                user_input=self.analyzed_input.original_query,
                content_query=self.analyzed_input.content,
                relevance_criteria=self.analyzed_input.relevance_criteria,
                anchor_doc_collection=self.anchor_doc_collection,
                extracted_name=self.analyzed_input.extracted_properties.specific_paper_name,
                recent_first=self.analyzed_input.extracted_properties.recent_first,
                recent_last=self.analyzed_input.extracted_properties.recent_last,
                central_first=self.analyzed_input.extracted_properties.central_first,
                central_last=self.analyzed_input.extracted_properties.central_last,
                suitable_for_by_citing=self.analyzed_input.extracted_properties.suitable_for_by_citing,
                time_range=self.analyzed_input.time_range,
                venues=self.venues,
                authors=self.authors,
                domains=self.analyzed_input.domains,
                doc_collection=self.input_doc_collection,
            )
        )

    async def run_fast_broad_search(self) -> OperativeResponse[BroadSearchOutput]:
        return await self.paper_finder_agent.fast_broad_search_agent(
            BroadSearchInput(
                user_input=self.analyzed_input.original_query,
                content_query=self.analyzed_input.content,
                relevance_criteria=self.analyzed_input.relevance_criteria,
                anchor_doc_collection=self.anchor_doc_collection,
                extracted_name=self.analyzed_input.extracted_properties.specific_paper_name,
                recent_first=self.analyzed_input.extracted_properties.recent_first,
                recent_last=self.analyzed_input.extracted_properties.recent_last,
                central_first=self.analyzed_input.extracted_properties.central_first,
                central_last=self.analyzed_input.extracted_properties.central_last,
                suitable_for_by_citing=self.analyzed_input.extracted_properties.suitable_for_by_citing,
                time_range=self.analyzed_input.time_range,
                venues=self.venues,
                authors=self.authors,
                domains=self.analyzed_input.domains,
                doc_collection=self.input_doc_collection,
            )
        )

    async def run_broad_search_by_keyword(self) -> OperativeResponse[BroadSearchByKeywordOutput]:
        return await self.paper_finder_agent.broad_search_by_keyword(
            BroadSearchByKeywordInput(
                user_input=self.analyzed_input.original_query,
                content_query=self.analyzed_input.content,
                relevance_criteria=self.analyzed_input.relevance_criteria,
                recent_first=self.analyzed_input.extracted_properties.recent_first,
                recent_last=self.analyzed_input.extracted_properties.recent_last,
                central_first=self.analyzed_input.extracted_properties.central_first,
                central_last=self.analyzed_input.extracted_properties.central_last,
                time_range=self.analyzed_input.time_range,
                venues=self.venues,
                authors=self.authors,
                domains=self.analyzed_input.domains,
                doc_collection=self.input_doc_collection,
            )
        )

    async def attempt_metadata_only_search(self) -> OperativeResponse[AgentOutput]:
        assert not self.analyzed_input.content
        self.consider_content_relevance = False
        return await self.paper_finder_agent.metadata_only_agent(
            MetadataOnlySearchInput(
                time_range=self.analyzed_input.time_range,
                venues=self.venues,
                domains=self.analyzed_input.domains,
                doc_collection=self.input_doc_collection,
            )
        )

    async def sort(self, docs: DocumentCollection) -> tuple[DocumentCollection, str]:
        if self.should_sort and docs.documents:
            analyzed_input = self.analyzed_input
            assume_recent_first = False
            if self.assume_recent_and_central_first and not analyzed_input.extracted_properties.recent_last:
                assume_recent_first = True
                logger.info("Assuming recent first.")
            assume_central_first = False
            if self.assume_recent_and_central_first and not analyzed_input.extracted_properties.central_last:
                assume_central_first = True
                logger.info("Assuming central first.")

            sorting_preferences = SortPreferences.from_analyzed_query(
                analyzed_input,
                assume_recent_first=assume_recent_first,
                assume_central_first=assume_central_first,
                consider_content_relevance=self.consider_content_relevance,
            )
            docs = await sorted_docs_by_preferences(docs, sorting_preferences)
            return docs, sorting_preferences.get_sorting_explanation()
        return docs, ""


async def get_paper_finder_output_from_docs(
    docs: DocumentCollection,
    aggregated_response_text: str,
    user_query: str,
    analyzed_input: AnalyzedQuery | PartiallyAnalyzedQuery,
    is_fast: bool,
    response_text_override: str | None = None,
) -> PaperFinderOutput:
    docs = await docs.with_fields([final_agent_score_field()])
    followup = ufv(uf.response_texts.suggest_broad_followup)
    if (
        analyzed_input.query_type
        and analyzed_input.query_type.broad_or_specific == "specific"
        and analyzed_input.content
    ):
        followup = ufv(uf.response_texts.suggest_to_search_for_papers_about_topic)
    return PaperFinderOutput(
        doc_collection=docs,
        response_text=join_paragraphs(
            generate_response_summary(
                docs,
                bool(analyzed_input.query_type) and analyzed_input.query_type.broad_or_specific == "broad",
                bool(analyzed_input.content),
            ),
            *(HRULE, ufv(uf.response_texts.detailed_response_prefix)),
            *explain_query_analysis(analyzed_input),
            aggregated_response_text,
            *(HRULE, followup),
            *(HRULE, ufv(uf.response_texts.work_harder_text))
            if (
                is_fast
                and isinstance(analyzed_input, AnalyzedQuery)
                and analyzed_input.query_type.type.startswith("BROAD")
            )
            else "",
        )
        if not response_text_override
        else response_text_override,
        input_query=user_query,
        analyzed_query=(
            analyzed_input
            if isinstance(analyzed_input, AnalyzedQuery)
            else analyzed_input.to_analyzed_query()
            if isinstance(analyzed_input, PartiallyAnalyzedQuery)
            else None
        ),
    )


@operative_session(PaperFinderAgent, "paper_finder_agent", allow_interactions=False)
async def run_agent(
    query: PaperFinderInput | InquiryReply, conversation_thread_id: ConversationThreadId
) -> OperativeResponse[PaperFinderOutput] | InquiryQuestion: ...
