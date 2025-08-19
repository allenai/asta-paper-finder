import logging
from typing import Coroutine, Literal

import pandas as pd
from ai2i.config import config_value
from ai2i.dcollection import BASIC_FIELDS, DocumentCollection, keyed_by_corpus_id
from ai2i.dcollection.fetchers.s2 import check_if_paper_inserted_before, get_publication_date_from_inserted_before
from ai2i.di import DI
from pydantic import Field

from mabool.agents.common.common import AgentState
from mabool.agents.common.relevance_judgement_utils import get_relevant_docs
from mabool.agents.common.utils import alog_args
from mabool.agents.complex_search.definitions import BroadSearchInput, BroadSearchOutput
from mabool.agents.snowball.snippet_snowball import run_snippet_snowball
from mabool.agents.snowball.snowball_utils import add_snowball_origins
from mabool.data_model.agent import AgentError
from mabool.data_model.config import cfg_schema
from mabool.data_model.rounds import RoundContext
from mabool.infra.operatives import (
    CompleteResponse,
    Operative,
    OperativeResponse,
    PartialResponse,
)
from mabool.utils import context_deps
from mabool.utils.asyncio import custom_gather
from mabool.utils.dc import DC

logger = logging.getLogger(__name__)


class SnowballState(AgentState):
    search_iteration: int = Field(default=1)


SnowballOutput = BroadSearchOutput

SnowballInput = BroadSearchInput


class SnowballExtendedInput(SnowballInput):
    forward_top_k: int = Field(
        default_factory=lambda: config_value(cfg_schema.snowball_agent.forward_top_k),
        ge=0,
        description="The number of top candidates to promote in forward snowball",
    )
    backward_top_k: int = Field(
        default_factory=lambda: config_value(cfg_schema.snowball_agent.backward_top_k),
        ge=0,
        description="The number of top candidates to promote in backward snowball",
    )
    snippet_top_k: int = Field(
        default_factory=lambda: config_value(cfg_schema.snowball_agent.snippet_top_k),
        ge=0,
        description="The number of top candidates to promote in snippet snowball",
    )
    search_iteration: int = Field(default=1)


async def run_paper_snowball(
    inputs: SnowballExtendedInput,
) -> DocumentCollection:
    try:
        seed_documents = inputs.doc_collection.filter(
            lambda doc: doc.relevance_judgement is not None and doc.relevance_judgement.relevance > 1
        )
        if len(seed_documents) == 0:
            logger.info("No relevant documents to snowball from.")
            return inputs.doc_collection
        directions_to_run: list[Coroutine] = []
        if inputs.forward_top_k:
            logger.info("=== Forward snowball ===")
            directions_to_run.append(do_forward_snowball(seed_documents, inputs))
        if inputs.backward_top_k:
            logger.info("=== Backward snowball ===")
            directions_to_run.append(do_backward_snowball(seed_documents, inputs))
        snowball_results = await custom_gather(*directions_to_run)
        return DC.merge(snowball_results)
    except Exception as e:
        logger.exception("An error occurred during snowballing: %s", e)
        return inputs.doc_collection


async def run_snowball(
    inputs: SnowballExtendedInput,
) -> DocumentCollection:
    snowball_results = await custom_gather(
        run_paper_snowball(inputs),
        run_snippet_snowball(
            content_query=inputs.content_query,
            doc_collection=inputs.doc_collection,
            top_k=inputs.snippet_top_k,
            search_iteration=inputs.search_iteration,
        ),
    )
    return DC.merge(snowball_results)


async def do_forward_snowball(
    seed_documents: DocumentCollection, snowball_config: SnowballExtendedInput
) -> DocumentCollection:
    logger.info("Adding citations from S2 API for relevant documents.")
    relevant_docs = await get_relevant_docs(seed_documents, threshold=1).with_fields(
        ["citations", "citation_count", "influential_citation_count"]
    )
    seed_for_forward_iteration = seed_documents.merged(relevant_docs)

    logger.info("Adding top candidates based on citation graph.")
    promoted_with_relevance = await promote_top_forward_candidates(seed_for_forward_iteration, snowball_config)
    return seed_documents.merged(promoted_with_relevance)


async def promote_top_forward_candidates(
    documents: DocumentCollection,
    snowball_config: SnowballExtendedInput,
) -> DocumentCollection:
    dff = get_forward_df(documents)
    if dff.empty:
        logger.info("No forward candidates found.")
        return documents

    forward_scores = get_forward_candidate_scores(dff)
    top_corpus_ids = (
        forward_scores.sort_values("score", ascending=False, kind="stable")
        .head(snowball_config.forward_top_k)
        .candidate_corpus_id.astype(str)
        .tolist()
    )
    promoted_docs = await DC.from_ids(top_corpus_ids).with_fields(BASIC_FIELDS)

    promoted_docs = add_snowball_origins(
        promoted_docs, variant="forward", search_iteration=snowball_config.search_iteration
    )

    return promoted_docs


@DI.managed
def get_forward_df(
    seed_document_collection: DocumentCollection,
    request_context: RoundContext | None = DI.requires(context_deps.request_context),
) -> pd.DataFrame:
    candidate_citations = []
    seed_docs = seed_document_collection.documents
    seed_docs_by_corpus_id = keyed_by_corpus_id(seed_docs)
    for seed_doc in seed_docs:
        if seed_doc.is_loaded("citations") and seed_doc.citations:
            for citation in seed_doc.citations:
                if request_context and not check_if_paper_inserted_before(
                    get_publication_date_from_inserted_before(request_context.inserted_before),
                    citation.year,
                    citation.publication_date,
                ):
                    logger.warning(f"skipping citation {citation.target_corpus_id} as its after insertion date")
                    continue
                if not seed_docs_by_corpus_id.get(str(citation.target_corpus_id)):
                    candidate_citations.append(
                        {
                            "candidate_corpus_id": citation.target_corpus_id,
                            "candidate_reference_count": citation.reference_count,
                            "seed_corpus_id": seed_doc.corpus_id,
                            "seed_relevance": seed_doc.relevance_judgement.relevance
                            if seed_doc.relevance_judgement
                            else 0,
                            "seed_citation_count": seed_doc.citation_count,
                            "is_influential": bool(citation.is_influential),
                            "num_contexts": citation.num_contexts or 1,
                        }
                    )

    dff = pd.DataFrame(candidate_citations)

    if dff.empty:
        return dff

    # fill missing values with median
    if "candidate_reference_count" not in dff or dff.candidate_reference_count.isnull().all():
        dff["candidate_reference_count"] = 1
    else:
        reference_count_default_value = dff.candidate_reference_count.median()
        dff["candidate_reference_count"] = (
            dff["candidate_reference_count"].fillna(reference_count_default_value).astype(int)
        )
    if "seed_citation_count" not in dff or dff.seed_citation_count.isnull().all():
        dff["seed_citation_count"] = 1
    else:
        citation_count_default_value = dff.seed_citation_count.median()
        dff["seed_citation_count"] = dff["seed_citation_count"].fillna(citation_count_default_value).astype(int)

    return dff


def get_forward_candidate_scores(dff: pd.DataFrame) -> pd.DataFrame:
    forward_scores = []
    for corpus_id, group in dff.groupby("candidate_corpus_id"):
        forward_scores.append(
            {
                "candidate_corpus_id": corpus_id,
                "score": score_forward_candidate(group),
            }
        )
    return pd.DataFrame(forward_scores)


def score_forward_candidate(
    citations: pd.DataFrame,
    seed_relevance_bias: float = 1,
    influential_bias: float = 0.1,
    num_contexts_bias: float = 0,
    candidate_citation_count_bias: float = -0.005,
    seed_citation_count_bias: float = 0,
) -> float:
    return score_snowball_candidate(
        citations,
        direction="forward",
        seed_relevance_bias=seed_relevance_bias,
        influential_bias=influential_bias,
        num_contexts_bias=num_contexts_bias,
        candidate_citation_count_bias=candidate_citation_count_bias,
        seed_citation_count_bias=seed_citation_count_bias,
    )


def get_backward_candidate_scores(dfb: pd.DataFrame) -> pd.DataFrame:
    backward_scores = []
    for corpus_id, group in dfb.groupby("candidate_corpus_id"):
        backward_scores.append(
            {
                "candidate_corpus_id": corpus_id,
                "score": score_backward_candidate(group),
            }
        )
    return pd.DataFrame(backward_scores)


def get_backward_df(seed_document_collection: DocumentCollection) -> pd.DataFrame:
    candidate_citations = []
    seed_docs = seed_document_collection.documents
    seed_docs_by_corpus_id = keyed_by_corpus_id(seed_docs)
    for seed_doc in seed_docs:
        if seed_doc.is_loaded("references") and seed_doc.references:
            for reference in seed_doc.references:
                if not seed_docs_by_corpus_id.get(str(reference.target_corpus_id)):
                    candidate_citations.append(
                        {
                            "candidate_corpus_id": reference.target_corpus_id,
                            "candidate_citation_count": reference.citation_count,
                            "seed_corpus_id": seed_doc.corpus_id,
                            "seed_relevance": seed_doc.relevance_judgement.relevance
                            if seed_doc.relevance_judgement
                            else 0,
                            "seed_reference_count": seed_doc.reference_count,
                            "is_influential": bool(reference.is_influential),
                            "num_contexts": reference.num_contexts or 1,
                        }
                    )

    dfb = pd.DataFrame(candidate_citations)

    if dfb.empty:
        return dfb

    # fill missing values with median
    if dfb.candidate_citation_count.isnull().all():
        dfb["candidate_citation_count"] = 1
    else:
        citation_count_default_value = dfb.candidate_citation_count.median()
        dfb["candidate_citation_count"] = (
            dfb["candidate_citation_count"].fillna(citation_count_default_value).astype(int)
        )
    if dfb.seed_reference_count.isnull().all():
        dfb["seed_reference_count"] = 1
    else:
        reference_count_default_value = dfb.seed_reference_count.median()
        dfb["seed_reference_count"] = dfb["seed_reference_count"].fillna(reference_count_default_value).astype(int)

    return dfb


async def do_backward_snowball(
    seed_documents: DocumentCollection, snowball_config: SnowballExtendedInput
) -> DocumentCollection:
    logger.info("Adding citations from S2 API for relevant documents.")
    relevant_docs = await get_relevant_docs(seed_documents, threshold=1).with_fields(
        ["references", "citation_count", "reference_count", "influential_citation_count"]
    )
    seed_for_backwards_iteration = seed_documents.merged(relevant_docs)

    logger.info("Adding top candidates based on citation graph.")
    promoted_docs = await promote_top_backward_candidates(seed_for_backwards_iteration, snowball_config)
    return seed_documents.merged(promoted_docs)


async def promote_top_backward_candidates(
    documents: DocumentCollection,
    snowball_config: SnowballExtendedInput,
) -> DocumentCollection:
    dfb = get_backward_df(documents)
    if dfb.empty:
        logger.info("No backward candidates found.")
        return documents

    backwards_scores = get_backward_candidate_scores(dfb)
    top_corpus_ids = (
        backwards_scores.sort_values("score", ascending=False, kind="stable")
        .head(snowball_config.backward_top_k)
        .candidate_corpus_id.astype(str)
        .tolist()
    )
    promoted_docs = await DC.from_ids(top_corpus_ids).with_fields(BASIC_FIELDS)

    promoted_docs = add_snowball_origins(
        promoted_docs, variant="backward", search_iteration=snowball_config.search_iteration
    )
    return promoted_docs


def score_backward_candidate(
    citations: pd.DataFrame,
    seed_relevance_bias: float = 1,
    influential_bias: float = 0,
    num_contexts_bias: float = 0,
    candidate_citation_count_bias: float = -0.0005,
    seed_citation_count_bias: float = 0,
) -> float:
    return score_snowball_candidate(
        citations,
        direction="backward",
        seed_relevance_bias=seed_relevance_bias,
        influential_bias=influential_bias,
        num_contexts_bias=num_contexts_bias,
        candidate_citation_count_bias=candidate_citation_count_bias,
        seed_citation_count_bias=seed_citation_count_bias,
    )


def score_snowball_candidate(
    citations: pd.DataFrame,
    direction: Literal["forward", "backward"],
    seed_relevance_bias: float,
    influential_bias: float,
    num_contexts_bias: float,
    candidate_citation_count_bias: float = 0,
    seed_citation_count_bias: float = 0,
) -> float:
    if direction == "backward":
        candidate_citation_count = max(citations.iloc[0]["candidate_citation_count"], len(citations))
    elif direction == "forward":
        candidate_citation_count = max(citations.iloc[0]["candidate_reference_count"], len(citations))
    score = 0
    for _, c in citations.iterrows():
        if direction == "backward":
            seed_citation_count = max(c["seed_reference_count"], 10)
        elif direction == "forward":
            seed_citation_count = max(c["seed_citation_count"], 1)

        score += (
            seed_relevance_bias * c["seed_relevance"]
            + influential_bias * int(c["is_influential"])
            + num_contexts_bias * c["num_contexts"]
            + (candidate_citation_count_bias * candidate_citation_count)
            + (seed_citation_count_bias * seed_citation_count)
        )
    return score


class SnowballAgent(Operative[SnowballExtendedInput, SnowballOutput, SnowballState]):
    def register(self) -> None: ...

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self, state: SnowballState | None, inputs: SnowballInput
    ) -> tuple[SnowballState | None, OperativeResponse[SnowballOutput]]:
        state = state or SnowballState(checkpoint=DC.empty())
        try:
            seed_collection = inputs.doc_collection.merged(inputs.anchor_doc_collection)
            if len(seed_collection) > 0:
                forward_only = (config_value(cfg_schema.run_snowball_for_recent) and inputs.recent_first) and not (
                    inputs.central_first or inputs.central_last
                )
                result_docs = await run_snowball(
                    SnowballExtendedInput(
                        user_input=inputs.user_input,
                        backward_top_k=0 if forward_only else config_value(cfg_schema.snowball_agent.backward_top_k),
                        forward_top_k=config_value(cfg_schema.snowball_agent.forward_top_k),
                        content_query=inputs.content_query,
                        relevance_criteria=inputs.relevance_criteria,
                        search_iteration=state.search_iteration,
                        doc_collection=inputs.doc_collection,
                        recent_first=inputs.recent_first,
                        recent_last=inputs.recent_last,
                        central_first=inputs.central_first,
                        central_last=inputs.central_last,
                        time_range=inputs.time_range,
                        venues=inputs.venues,
                        authors=inputs.authors,
                        domains=inputs.domains,
                    )
                )
            else:
                result_docs = DC.empty()
            operative_response = CompleteResponse(
                data=SnowballOutput(
                    doc_collection=result_docs,
                )
            )
            return (
                SnowballState(checkpoint=result_docs, search_iteration=state.search_iteration + 1),
                operative_response,
            )
        except Exception as e:
            logger.exception("An error occurred during snowballing: %s", e)
            result_docs = inputs.doc_collection
            partial_response = PartialResponse(
                data=SnowballOutput(
                    doc_collection=result_docs,
                ),
                error=AgentError(
                    type="other",
                    message="Results are partial, as an error occurred during the process.",
                ),
            )
            return SnowballState(checkpoint=result_docs, search_iteration=state.search_iteration + 1), partial_response
