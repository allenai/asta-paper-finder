from typing import Any, Iterable, Sequence

from ai2i.dcollection import (
    CorpusId,
    DenseDataset,
    Document,
    DocumentCollection,
    DocumentCollectionFactory,
    DocumentFieldName,
    ExtractedYearlyTimeRange,
)
from ai2i.di import DI

from mabool.data_model.rounds import RoundContext
from mabool.utils import context_deps, dc_deps


class DC:
    @staticmethod
    @DI.managed
    def from_ids(
        corpus_ids: list[CorpusId], dcf: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory)
    ) -> DocumentCollection:
        return dcf.from_ids(corpus_ids)

    @staticmethod
    @DI.managed
    def from_docs(
        documents: Sequence[Document],
        computed_fields: dict[DocumentFieldName, Any] | None = None,
        dcf: DocumentCollectionFactory = DI.requires(
            dc_deps.round_doc_collection_factory, default_factory=dc_deps.detached_doc_collection_factory
        ),
    ) -> DocumentCollection:
        return dcf.from_docs(documents, computed_fields)

    @staticmethod
    @DI.managed
    def empty(dcf: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory)) -> DocumentCollection:
        return dcf.empty()

    @staticmethod
    @DI.managed
    def merge(
        collections: Iterable[DocumentCollection],
        dcf: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory),
    ) -> DocumentCollection:
        return dcf.merge(collections)

    @staticmethod
    @DI.managed
    async def from_s2_by_author(
        authors_profiles: list[list[Any]],
        limit: int,
        dcf: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory),
        request_context: RoundContext | None = DI.requires(context_deps.request_context),
    ) -> DocumentCollection:
        return await dcf.from_s2_by_author(
            authors_profiles, limit, request_context.inserted_before if request_context else None
        )

    @staticmethod
    @DI.managed
    async def from_s2_by_title(
        query: str,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        dcf: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory),
        request_context: RoundContext | None = DI.requires(context_deps.request_context),
    ) -> DocumentCollection:
        return await dcf.from_s2_by_title(
            query, time_range, venues, request_context.inserted_before if request_context else None
        )

    @staticmethod
    @DI.managed
    async def from_s2_search(
        query: str,
        limit: int,
        search_iteration: int = 1,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        fields_of_study: list[str] | None = None,
        fields: list[DocumentFieldName] | None = None,
        dcf: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory),
        request_context: RoundContext | None = DI.requires(context_deps.request_context),
    ) -> DocumentCollection:
        return await dcf.from_s2_search(
            query,
            limit,
            search_iteration,
            time_range,
            venues,
            fields_of_study,
            None,
            fields,
            request_context.inserted_before if request_context else None,
        )

    @staticmethod
    @DI.managed
    async def from_s2_citing_papers(
        corpus_id: CorpusId,
        search_iteration: int = 1,
        dcf: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory),
        request_context: RoundContext | None = DI.requires(context_deps.request_context),
    ) -> DocumentCollection:
        return await dcf.from_s2_citing_papers(
            corpus_id, search_iteration, inserted_before=request_context.inserted_before if request_context else None
        )

    @staticmethod
    @DI.managed
    async def from_dense_retrieval(
        queries: list[str],
        search_iteration: int,
        dataset: DenseDataset,
        top_k: int,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        authors: list[str] | None = None,
        corpus_ids: list[CorpusId] | None = None,
        fields_of_study: list[str] | None = None,
        dcf: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory),
        request_context: RoundContext | None = DI.requires(context_deps.request_context),
    ) -> DocumentCollection:
        return await dcf.from_dense_retrieval(
            queries,
            search_iteration,
            dataset,
            top_k,
            time_range,
            venues,
            authors,
            corpus_ids,
            fields_of_study,
            request_context.inserted_before if request_context else None,
        )
