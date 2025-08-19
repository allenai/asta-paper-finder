from __future__ import annotations

import logging

from ai2i.chain import ModelName
from ai2i.dcollection import CorpusId, DenseDataset, DocumentCollection, ExtractedYearlyTimeRange
from ai2i.di import DI
from pydantic import BaseModel, Field

from mabool.agents.common.common import AgentState
from mabool.agents.common.domain_utils import get_dense_datasets_by_domains, get_fields_of_study_filter_from_domains
from mabool.agents.common.utils import alog_args
from mabool.agents.complex_search.definitions import BroadSearchInput, BroadSearchOutput
from mabool.agents.dense.formulation import DenseQuery, get_reformulated_dense_queries
from mabool.data_model.agent import AgentError, DomainsIdentified
from mabool.data_model.config import cfg_schema
from mabool.infra.operatives import (
    CompleteResponse,
    Operative,
    OperativeResponse,
    VoidResponse,
)
from mabool.utils import dc_deps
from mabool.utils.asyncio import custom_gather
from mabool.utils.dc import DC

logger = logging.getLogger(__name__)


@DI.managed
async def run_dense_agent(
    search_query: str,
    domains: DomainsIdentified,
    search_iteration: int,
    documents: DocumentCollection,
    dense_queries: list[DenseQuery],
    exclude_corpus_ids: set[CorpusId],
    anchor_doc_collection: DocumentCollection = DI.requires(dc_deps.empty_doc_collection),
    time_range: ExtractedYearlyTimeRange | None = None,
    venues: list[str] | None = None,
    authors: list[str] | None = None,
) -> DenseIterationResult:
    dense_iteration_result = await fetch_documents_from_dense_indices(
        search_query,
        domains,
        search_iteration,
        documents,
        dense_queries,
        exclude_corpus_ids,
        anchor_doc_collection,
        time_range,
        authors,
        venues,
    )

    return DenseIterationResult(
        documents=dense_iteration_result.documents,
        dense_queries=dense_iteration_result.dense_queries,
        exclude_corpus_ids=dense_iteration_result.exclude_corpus_ids,
    )


def get_queries_per_dataset(iteration_dense_queries: list[str], dataset: DenseDataset) -> list[str]:
    # take minority to run on all slow dense indexes (e.g. vespa)
    return (
        iteration_dense_queries[
            : int(len(iteration_dense_queries) * 0.2)
        ]  # first should be the original extracted content
        + iteration_dense_queries[int(len(iteration_dense_queries) * 0.8) :]  # these should be alternative/reformulated
    )


@DI.managed
async def fetch_documents_from_dense_indices(
    search_query: str,
    domains: DomainsIdentified,
    search_iteration: int,
    documents: DocumentCollection,
    dense_queries: list[DenseQuery],
    exclude_corpus_ids: set[CorpusId],
    anchor_doc_collection: DocumentCollection = DI.requires(dc_deps.empty_doc_collection),
    time_range: ExtractedYearlyTimeRange | None = None,
    venues: list[str] | None = None,
    authors: list[str] | None = None,
    initial_top_k_per_query: int = DI.config(cfg_schema.dense_agent.initial_top_k_per_query),
    formulation_model_name: ModelName = DI.config(cfg_schema.dense_agent.formulation_model_name),
    reformulate_prompt_example_docs: int = DI.config(cfg_schema.dense_agent.reformulate_prompt_example_docs),
    reformulate_prompt_num_queries: int = DI.config(cfg_schema.dense_agent.reformulate_prompt_num_queries),
) -> DenseIterationResult:
    dense_datasets = get_dense_datasets_by_domains(domains)
    fields_of_study = get_fields_of_study_filter_from_domains(domains)

    logger.info("Formulating dense queries")
    iteration_dense_queries, used_corpus_ids = await get_reformulated_dense_queries(
        search_query,
        documents=documents,
        model_name=formulation_model_name,
        anchor_doc_collection=anchor_doc_collection,
        max_docs_in_prompt=reformulate_prompt_example_docs,
        exclude_corpus_ids=exclude_corpus_ids,
        relevance_threshold=1,
        max_queries_to_generate=reformulate_prompt_num_queries,
        use_search_query_as_one_of_the_queries=search_iteration == 1 and len(search_query.split()) <= 50,
    )
    logger.info("Dense queries:\n - " + "\n - ".join(iteration_dense_queries))
    exclude_corpus_ids.update(used_corpus_ids)
    logger.info(f"Fetching {initial_top_k_per_query} documents for each dense query")
    iteration_dense_results = await custom_gather(
        *(
            DC.from_dense_retrieval(
                queries=get_queries_per_dataset(iteration_dense_queries, dataset),
                search_iteration=search_iteration,
                dataset=dataset,
                top_k=initial_top_k_per_query,
                time_range=time_range,
                venues=venues,
                authors=authors,
                fields_of_study=fields_of_study,
            )
            for dataset in dense_datasets
        ),
    )
    iteration_documents = DC.merge(iteration_dense_results)
    dense_queries.extend(iteration_dense_queries)
    documents = documents.merged(iteration_documents)
    return DenseIterationResult(documents=documents, dense_queries=dense_queries, exclude_corpus_ids=exclude_corpus_ids)


class DenseState(AgentState):
    dense_queries: list[DenseQuery] = Field(default_factory=list)
    exclude_corpus_ids: set[CorpusId] = Field(default_factory=set)
    search_iteration: int = Field(default=1)


DenseInput = BroadSearchInput
DenseOutput = BroadSearchOutput


class DenseIterationResult(BaseModel):
    documents: DocumentCollection = Field(default_factory=DC.empty)
    dense_queries: list[DenseQuery] = Field(default_factory=list)
    exclude_corpus_ids: set[CorpusId] = Field(default_factory=set)


class DenseAgent(Operative[DenseInput, DenseOutput, DenseState]):
    def register(self) -> None: ...

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self, state: DenseState | None, inputs: DenseInput
    ) -> tuple[DenseState | None, OperativeResponse[DenseOutput]]:
        try:
            state = state or DenseState(checkpoint=DC.empty())
            docs, dense_queries, exclude_corpus_ids = (
                state.checkpoint,
                state.dense_queries,
                state.exclude_corpus_ids,
            )
            if not docs:
                documents = inputs.doc_collection
            else:
                documents = docs.merged(inputs.doc_collection)

            dense_result = await run_dense_agent(
                search_query=inputs.content_query,
                domains=inputs.domains,
                search_iteration=state.search_iteration,
                documents=documents or DC.empty(),
                dense_queries=dense_queries,
                exclude_corpus_ids=exclude_corpus_ids,
                anchor_doc_collection=inputs.anchor_doc_collection,
                time_range=inputs.time_range,
                venues=inputs.venues,
                authors=inputs.authors,
            )
            operative_response = CompleteResponse(
                data=DenseOutput(
                    doc_collection=dense_result.documents,
                )
            )
            return (
                DenseState(
                    checkpoint=dense_result.documents,
                    dense_queries=dense_result.dense_queries,
                    exclude_corpus_ids=dense_result.exclude_corpus_ids,
                    search_iteration=state.search_iteration + 1,
                ),
                operative_response,
            )
        except Exception as e:
            logger.exception(e)
            return None, VoidResponse(error=AgentError(type="other", message=str(e)))
