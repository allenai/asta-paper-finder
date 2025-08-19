from __future__ import annotations

import logging
from typing import Literal, NamedTuple

from ai2i.dcollection import PaperFinderDocument
from ai2i.dcollection.data_access_context import DocumentCollectionContext
from ai2i.dcollection.interface.collection import BASIC_FIELDS, Document
from ai2i.dcollection.interface.document import (
    CorpusId,
    DocumentFieldName,
    ExtractedYearlyTimeRange,
    OriginQuery,
    RefMention,
    SentenceOffsets,
    SimilarityScore,
    Snippet,
)

logger = logging.getLogger(__name__)


class DenseDataset(NamedTuple):
    provider: Literal["vespa"]
    name: Literal["open-nora"]
    variant: Literal["", "pa1-v1"]


async def fetch_from_vespa_dense_retrieval(
    queries: list[str],
    search_iteration: int,
    fields: list[DocumentFieldName],
    top_k: int,
    dataset: DenseDataset,
    context: DocumentCollectionContext,
    time_range: ExtractedYearlyTimeRange | None = None,
    venues: list[str] | None = None,
    authors: list[str] | None = None,
    corpus_ids: list[CorpusId] | None = None,
    fields_of_study: list[str] | None = None,
    vespa_concurrency: int = 10,
    inserted_before: str | None = None,
) -> list[Document]:
    fields = fields.copy()
    fields.append("corpus_id")
    dense_results = await context.vespa_client.abatch(
        queries,
        top_k=top_k,
        time_range=time_range,
        venues=venues,
        authors=authors,
        corpus_ids=corpus_ids,
        fields_of_study=fields_of_study,
        config={
            "max_concurrency": vespa_concurrency,
        },
        inserted_before=inserted_before,
    )
    if all([len(dense_results_for_query) == 0 for dense_results_for_query in dense_results]):
        return []

    if dataset.variant != context.vespa_client.get_actual_vespa_version():
        logger.warning(
            f"inconsistency between expected ({dataset.variant}) and actual \
            ({context.vespa_client.get_actual_vespa_version()}) vespa index versions"
        )

    dense_results_with_queries = zip(dense_results, queries)

    docs = []
    basic_fields_to_load = [f for f in fields if f in BASIC_FIELDS]
    for dense_results_for_query, query in dense_results_with_queries:
        for i, dr in enumerate(dense_results_for_query):
            doc_id_fields = {k: dr.metadata["metadata"].get(k) for k in basic_fields_to_load}
            if doc_id_fields["corpus_id"] is None:
                logger.warning(f"Document fetched from vespa has no corpus_id: {doc_id_fields}")
                continue
            snippet: Snippet | None = None
            if "snippets" in fields:
                snippet = Snippet(
                    text=dr.page_content,
                    section_title=dr.metadata["metadata"].get("section_title"),
                    section_kind=dr.metadata["metadata"].get("section_kind"),
                    ref_mentions=[RefMention(**rm) for rm in dr.metadata["metadata"].get("ref_mentions", [])],
                    sentences=[SentenceOffsets(**sso) for sso in dr.metadata["metadata"].get("sentence_offsets", [])],
                    char_start_offset=dr.metadata["sentence"]["document_char_offsets"][0],
                    char_end_offset=dr.metadata["sentence"]["document_char_offsets"][1],
                    bounding_boxes=dr.metadata["metadata"]["bounding_boxes"],
                    similarity_scores=[
                        SimilarityScore(
                            query=query,
                            similarity_model_name=repr(dataset),
                            score=float(str(dr.metadata.get("dense_similarity"))),
                        )
                    ],
                )
            doc = PaperFinderDocument(
                corpus_id=str(doc_id_fields["corpus_id"]),
                url=doc_id_fields["url"],
                title=doc_id_fields["title"],
                year=doc_id_fields["year"],
                authors=doc_id_fields["authors"],
                abstract=doc_id_fields["abstract"],
                venue=doc_id_fields["venue"],
                snippets=[snippet] if snippet else [],
                origins=[
                    OriginQuery(
                        query_type="dense",
                        provider=dataset.provider,
                        dataset=dataset.name,
                        variant=context.vespa_client.get_actual_vespa_version(),
                        query=query,
                        iteration=search_iteration,
                        ranks=[i + 1],
                    )
                ],
            )
            docs.append(doc)
    return docs
