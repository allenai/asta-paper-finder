import logging
from typing import Sequence

from ai2i.chain import LLMModel, ModelName, Timeouts, define_llm_endpoint
from ai2i.dcollection import CorpusId, DocumentCollection, SampleMethod
from ai2i.di import DI

from mabool.agents.common.relevance_judgement_utils import get_relevant_docs
from mabool.agents.dense.formulation_prompts import (
    dense_formulate_prompts,
    dense_reformulate,
)
from mabool.utils import dc_deps
from mabool.utils.asyncio import custom_gather
from mabool.utils.llm_utils import get_api_key_for_model

logger = logging.getLogger(__name__)

DenseQuery = str


@DI.managed
async def get_reformulated_dense_queries(
    search_query: str,
    documents: DocumentCollection,
    model_name: ModelName,
    anchor_doc_collection: DocumentCollection = DI.requires(dc_deps.empty_doc_collection),
    max_docs_in_prompt: int = 10,
    doc_sampling_method: SampleMethod = "bottom_origin_rank_stratified_relevance",
    max_queries_to_generate: int = 10,
    exclude_corpus_ids: set[str] | None = None,
    relevance_threshold: int = 2,
    use_search_query_as_one_of_the_queries: bool = False,
) -> tuple[list[DenseQuery], list[CorpusId]]:
    if use_search_query_as_one_of_the_queries:
        max_queries_to_generate -= 1

    llm_model = LLMModel.from_name(model_name)
    endpoint = define_llm_endpoint(
        default_timeout=Timeouts.medium,
        default_model=llm_model,
        logger=logger,
        api_key=get_api_key_for_model(llm_model),
    )

    if len(documents) == 0 and len(anchor_doc_collection) == 0:
        num_formulate_prompts = len(dense_formulate_prompts)
        # divide the max_queries_to_generate between the different formulation prompts
        # if there is a remainder, distribute it between the prompts
        max_output_for_each_prompt: list[int] = [
            max_queries_to_generate // num_formulate_prompts
        ] * num_formulate_prompts
        for i in range(max_queries_to_generate % num_formulate_prompts):
            max_output_for_each_prompt[i] += 1
        dense_formulate_coroutines = [
            endpoint.execute(dense_formulate).once(
                {"search_query": search_query, "max_output": max_output_for_each_prompt[i]}
            )
            for i, dense_formulate in enumerate(dense_formulate_prompts)
        ]
        formulate_results: Sequence[list[str] | BaseException] = await custom_gather(
            *dense_formulate_coroutines, return_exceptions=True
        )
        dense_queries: list[str] = []
        for result in formulate_results:
            if isinstance(result, BaseException):
                logger.error(f"Error while formulating dense queries: {result}")
            else:
                dense_queries.extend(result)

        if use_search_query_as_one_of_the_queries:
            dense_queries = [search_query] + dense_queries
        return dense_queries, []
    else:
        relevant_docs = get_relevant_docs(documents, relevance_threshold)
        if exclude_corpus_ids:
            logger.info(f"Excluding {len(exclude_corpus_ids)} corpus ids from reformulation prompt.")
            relevant_docs = relevant_docs.filter(lambda doc: doc.corpus_id not in exclude_corpus_ids)

        logger.info(
            f"Found {len(relevant_docs)} relevant documents to use as examples for reformulation \
            (relevance>{relevance_threshold}), and {len(anchor_doc_collection)} anchor documents."
        )

        if len(relevant_docs) > 0 or len(anchor_doc_collection) > 0:
            relevant_docs = anchor_doc_collection + relevant_docs.sample(
                max_docs_in_prompt - len(anchor_doc_collection), method=doc_sampling_method
            )

            logger.info(f"Reformulating dense query based on top {len(relevant_docs)} relevant documents.")
            highly_relevant_doc_texts = "\n\n".join(doc.markdown or "" for doc in relevant_docs.documents)
            reformulated_queries = await endpoint.execute(dense_reformulate).once(
                {
                    "search_query": search_query,
                    "papers": highly_relevant_doc_texts,
                    "max_output": max_queries_to_generate,
                }
            )
            if use_search_query_as_one_of_the_queries:
                reformulated_queries = [search_query] + reformulated_queries
            return reformulated_queries, [doc.corpus_id for doc in relevant_docs.documents]
        else:
            logger.warning("No relevant documents found for reformulation.")
            return [], []
