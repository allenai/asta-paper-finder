from __future__ import annotations

import logging

from ai2i.common.utils.time import atiming
from ai2i.config import config_value
from ai2i.dcollection import (
    AssignedField,
    Document,
    DocumentCollection,
    DocumentCollectionSortDef,
    ExtractedYearlyTimeRange,
)
from ai2i.di import DI

from mabool.agents.common.common import AgentState, filter_docs_by_metadata
from mabool.agents.common.computed_fields.fields import rerank_score_field
from mabool.agents.common.computed_fields.relevance import relevance_judgement_field
from mabool.agents.common.domain_utils import get_dense_datasets_by_domains, get_fields_of_study_filter_from_domains
from mabool.agents.common.relevance_judgement_utils import (
    get_relevant_docs,
    log_relevance_value_counts,
    report_relevance_judgement_counts,
)
from mabool.agents.common.utils import alog_args
from mabool.agents.complex_search.definitions import BroadSearchInput, BroadSearchOutput
from mabool.agents.snowball.snippet_snowball import run_snippet_snowball
from mabool.data_model.agent import AgentError, DomainsIdentified, RelevanceCriteria
from mabool.data_model.config import cfg_schema
from mabool.infra.operatives import (
    CompleteResponse,
    Operative,
    OperativeResponse,
    VoidResponse,
)
from mabool.utils.asyncio import custom_gather
from mabool.utils.dc import DC

logger = logging.getLogger(__name__)

type FastBroadSearchState = AgentState


async def fast_broad_search(
    content_query: str,
    domains: DomainsIdentified,
    relevance_criteria: RelevanceCriteria,
    authors: list[str] | None = None,
    venues: list[str] | None = None,
    time_range: ExtractedYearlyTimeRange | None = None,
    cohere_rerank: bool = True,
) -> DocumentCollection:
    doc_collection = await _initial_retrieval(
        content_query, domains, authors=authors, venues=venues, time_range=time_range
    )
    doc_collection = await _rerank_docs(doc_collection, relevance_criteria, cohere_rerank)
    doc_collection = await _run_relevance_judgement(doc_collection, relevance_criteria)

    return doc_collection


@DI.managed
async def _run_initial_retrieval(
    content_query: str,
    domains: DomainsIdentified,
    authors: list[str] | None = None,
    venues: list[str] | None = None,
    time_range: ExtractedYearlyTimeRange | None = None,
) -> DocumentCollection:
    dense_datasets = get_dense_datasets_by_domains(domains)
    fields_of_study = get_fields_of_study_filter_from_domains(domains)

    dense_top_k = (
        config_value(cfg_schema.fast_broad_search_agent.dense_top_k)
        if len(dense_datasets) > 1
        else config_value(cfg_schema.fast_broad_search_agent.dense_top_k) * 2
    )

    retrieval_futures = [
        # step reporting in gather below
        DC.from_dense_retrieval(
            queries=[content_query],
            search_iteration=1,
            top_k=dense_top_k,
            dataset=dataset,
            authors=authors,
            venues=venues,
            time_range=time_range,
            fields_of_study=fields_of_study,
        )
        for dataset in dense_datasets
    ]
    retrieval_futures += [
        # step reporting in gather below
        DC.from_s2_search(
            content_query,
            limit=config_value(cfg_schema.fast_broad_search_agent.s2_relevance_search_top_k),
            search_iteration=1,
            venues=venues,
            time_range=time_range,
            fields_of_study=fields_of_study,
        )
    ]
    doc_collections_or_errors = await custom_gather(*retrieval_futures, return_exceptions=True)
    for dc_or_e in doc_collections_or_errors:
        if isinstance(dc_or_e, Exception):
            logger.error(f"Error in retrieval: {dc_or_e}", exc_info=dc_or_e)

    doc_collection = DC.merge(dc for dc in doc_collections_or_errors if not isinstance(dc, BaseException))
    if not doc_collection:
        logger.error("No documents retrieved from dense and s2 search")
        raise Exception("No documents retrieved from dense and s2 search")

    doc_collection += await run_snippet_snowball(
        content_query,
        doc_collection,
        top_k=config_value(cfg_schema.fast_broad_search_agent.snowball_snippets_top_k),
        search_iteration=1,
        fast_mode=True,
    )

    doc_collection = await filter_docs_by_metadata(
        doc_collection, authors=authors, venues=venues, time_range=time_range
    )

    return doc_collection


@atiming
async def _initial_retrieval(
    content_query: str,
    domains: DomainsIdentified,
    authors: list[str] | None = None,
    venues: list[str] | None = None,
    time_range: ExtractedYearlyTimeRange | None = None,
) -> DocumentCollection:
    # forwarding to a separate function because @app_ctx.managed and @traceable don't work well together.
    return await _run_initial_retrieval(content_query, domains, authors, venues, time_range)


@atiming
async def _rerank_docs(
    doc_collection: DocumentCollection, relevance_criteria: RelevanceCriteria | None = None, cohere_rerank: bool = True
) -> DocumentCollection:
    if cohere_rerank:
        try:
            doc_collection = await doc_collection.with_fields([rerank_score_field(relevance_criteria)])
        except Exception:
            cohere_rerank = False
        if cohere_rerank and any(doc.rerank_score is None for doc in doc_collection.documents):
            logger.warning("Failed to load rerank scores")
            cohere_rerank = False

        if cohere_rerank:
            doc_collection = doc_collection.sorted(
                sort_definitions=[DocumentCollectionSortDef(field_name="rerank_score", order="desc")]
            )
            return doc_collection

    if not cohere_rerank:
        logger.warning("Sorting by origin ranks")
        return await _sorted_by_origin_ranks(doc_collection)


async def _sorted_by_origin_ranks(doc_collection: DocumentCollection) -> DocumentCollection:
    # TODO: should probably prioritize/weight by origin (e.g. vespa > e5 > e5-abst-desc > snippets > s2)
    def _doc_best_rank(doc: Document) -> int | None:
        if not doc.origins:
            return None
        return min([min(origin.ranks) for origin in doc.origins if origin.ranks])

    best_ranks = [_doc_best_rank(doc) for doc in doc_collection.documents]
    if all(rank is None for rank in best_ranks):
        return doc_collection

    rank_for_docs_without_ranks = max(rank for rank in best_ranks if rank is not None) + 1
    best_ranks = [rank if rank is not None else rank_for_docs_without_ranks for rank in best_ranks]

    doc_collection = await doc_collection.with_fields(
        [
            AssignedField[int](
                field_name="best_origin_rank",
                required_fields=[],
                assigned_values=best_ranks,
            )
        ]
    )
    doc_collection = doc_collection.sorted(
        sort_definitions=[DocumentCollectionSortDef(field_name="best_origin_rank", order="asc")]
    )
    return doc_collection


@atiming
async def _run_relevance_judgement(
    doc_collection: DocumentCollection, relevance_criteria: RelevanceCriteria
) -> DocumentCollection:
    doc_collection = doc_collection.update_computed_fields([relevance_judgement_field(relevance_criteria)])
    doc_collection_with_rj = doc_collection.take(
        config_value(cfg_schema.fast_broad_search_agent.relevance_judgement_quota)
    )
    doc_collection_with_rj = await doc_collection_with_rj.with_fields(["relevance_judgement"])
    doc_collection = doc_collection_with_rj.merged(doc_collection)
    await log_relevance_value_counts(doc_collection)
    await report_relevance_judgement_counts(doc_collection)
    return doc_collection


class FastBroadSearchAgent(Operative[BroadSearchInput, BroadSearchOutput, FastBroadSearchState]):
    """
    FastBroadSearchAgent is an agent that performs a broad search using dense retrieval and snowballing.
    """

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self, state: FastBroadSearchState | None, inputs: BroadSearchInput
    ) -> tuple[FastBroadSearchState | None, OperativeResponse[BroadSearchOutput]]:
        try:
            doc_collection = await fast_broad_search(
                inputs.content_query,
                inputs.domains,
                relevance_criteria=inputs.relevance_criteria,
                authors=inputs.authors,
                venues=inputs.venues,
                time_range=inputs.time_range,
            )
            doc_collection = get_relevant_docs(doc_collection)

            return AgentState(checkpoint=doc_collection), CompleteResponse(
                data=BroadSearchOutput(
                    doc_collection=doc_collection,
                )
            )
        except Exception as e:
            logger.exception(f"An error occurred while running {self.__class__.__name__}: {e}")
            return None, VoidResponse(
                error=AgentError(type="other", message=f"BroadSearchAgent failed to respond; {str(e)}")
            )
