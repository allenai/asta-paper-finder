from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Iterable, Sequence

from semanticscholar import AsyncSemanticScholar

from ai2i.dcollection import PaperFinderDocument
from ai2i.dcollection.caching.cache import SubsetCache
from ai2i.dcollection.collection import PaperFinderDocumentCollection
from ai2i.dcollection.data_access_context import DocumentCollectionContext, SubsetCacheInterface
from ai2i.dcollection.external_api.dense.vespa import VespaRetriever
from ai2i.dcollection.fetchers.dense import DenseDataset, fetch_from_vespa_dense_retrieval
from ai2i.dcollection.fetchers.s2 import (
    s2_by_author,
    s2_fetch_citing_papers,
    s2_paper_search,
    s2_papers_by_title,
)
from ai2i.dcollection.interface.collection import (
    BASIC_FIELDS,
    S2_FIELDS,
    BaseDocumentCollectionFactory,
    Document,
    DocumentCollection,
)
from ai2i.dcollection.interface.document import (
    CorpusId,
    DocumentFieldName,
    ExtractedYearlyTimeRange,
)

logger = logging.getLogger(__name__)


class DocumentCollectionFactory(BaseDocumentCollectionFactory):
    def __init__(
        self,
        /,
        s2_api_key: str | None = None,
        s2_api_timeout: int = 60,
        cache_ttl: int = 600,
        cache_is_enabled: bool = True,
        force_deterministic: bool = False,
    ):
        super().__init__()
        s2_client = (
            AsyncSemanticScholar(timeout=s2_api_timeout, api_key=s2_api_key)
            if s2_api_key
            else AsyncSemanticScholar(timeout=s2_api_timeout)  # TODO: do we want to allow / warn in this case?
        )
        vespa_client = VespaRetriever(
            s2_client=s2_client,
            timeout=s2_api_timeout,
        )
        cache = SubsetCache(
            ttl=cache_ttl,
            is_enabled=cache_is_enabled,
            force_deterministic=force_deterministic,
        )
        self._context = DocumentCollectionContext(
            s2_client=s2_client,
            vespa_client=vespa_client,
            cache=cache,
            force_deterministic=force_deterministic,
        )

    def s2_client(self) -> AsyncSemanticScholar:
        return self._context.s2_client

    def cache(self) -> SubsetCacheInterface:
        return self._context.cache

    def context(self) -> DocumentCollectionContext:
        return self._context

    def from_ids(self, corpus_ids: list[CorpusId]) -> DocumentCollection:
        """Create a document collection from a list of corpus IDs."""
        return self.from_docs([PaperFinderDocument(corpus_id=corpus_id) for corpus_id in corpus_ids])

    def from_docs(
        self,
        documents: Sequence[Document],
        computed_fields: dict[DocumentFieldName, Any] | None = None,
    ) -> DocumentCollection:
        """Create a document collection from a list of documents."""
        docs_for_fuse: dict[CorpusId, list[Document]] = defaultdict(list)
        for doc in documents:
            docs_for_fuse[doc.corpus_id].append(doc)
        fused_docs = []
        for doc_group in docs_for_fuse.values():
            fused_docs.append(doc_group[0].fuse(*doc_group[1:]))
        return (
            PaperFinderDocumentCollection(documents=list(fused_docs), factory=self)
            if not computed_fields
            else PaperFinderDocumentCollection(
                documents=list(fused_docs),
                computed_fields=computed_fields or {},
                factory=self,
            )
        )

    def empty(self) -> DocumentCollection:
        """Create an empty document collection."""
        return self.from_docs([])

    def merge(self, collections: Iterable[DocumentCollection]) -> DocumentCollection:
        return PaperFinderDocumentCollection(factory=self).merged(*collections)

    def from_dict(self, params: dict[str, Any]) -> PaperFinderDocumentCollection:
        rest = {k: v for k, v in params.items() if k != "documents"}
        if "documents" in params:
            documents = [PaperFinderDocument.from_dict(d) for d in params["documents"]]
        else:
            documents = []
        return PaperFinderDocumentCollection(documents=documents, factory=self, **rest)

    async def from_s2_by_author(
        self, authors_profiles: list[list[Any]], limit: int, inserted_before: str | None
    ) -> DocumentCollection:
        """Create a document collection from S2 by author."""
        context = self._context
        if len(authors_profiles) == 1:
            documents = await s2_by_author(authors_profiles[0], context, inserted_before)
        else:
            raise NotImplementedError

        collection = self.from_docs(documents=documents)
        return await collection.take(limit).with_fields(BASIC_FIELDS)

    async def from_s2_by_title(
        self,
        query: str,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        inserted_before: str | None = None,
    ) -> DocumentCollection:
        """Create a document collection from S2 by title."""
        documents = await s2_papers_by_title(
            query,
            time_range=time_range,
            venues=venues,
            context=self._context,
            inserted_before=inserted_before,
        )
        collection = self.from_docs(documents=documents)
        return await collection.with_fields(BASIC_FIELDS)

    async def from_s2_search(
        self,
        query: str,
        limit: int,
        search_iteration: int = 1,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        fields_of_study: list[str] | None = None,
        min_citations: int | None = None,
        fields: list[DocumentFieldName] | None = None,
        inserted_before: str | None = None,
    ) -> DocumentCollection:
        """Create a document collection from S2 search."""
        documents = await s2_paper_search(
            query,
            search_iteration=search_iteration,
            time_range=time_range,
            venues=venues,
            fields_of_study=fields_of_study,
            min_citations=min_citations,
            total_limit=limit,
            context=self._context,
            inserted_before=inserted_before,
        )
        collection = self.from_docs(documents=documents)
        return await collection.with_fields(fields or BASIC_FIELDS)

    async def from_s2_citing_papers(
        self,
        corpus_id: CorpusId,
        search_iteration: int = 1,
        total_limit: int = 1000,
        inserted_before: str | None = None,
    ) -> DocumentCollection:
        """Create a document collection from S2 citing papers."""
        documents = await s2_fetch_citing_papers(
            corpus_id,
            search_iteration=search_iteration,
            context=self._context,
            total_limit=total_limit,
            inserted_before=inserted_before,
        )
        collection = self.from_docs(documents=documents)
        return await collection.with_fields(BASIC_FIELDS)

    async def from_dense_retrieval(
        self,
        queries: list[str],
        search_iteration: int,
        dataset: DenseDataset,
        top_k: int,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        authors: list[str] | None = None,
        corpus_ids: list[CorpusId] | None = None,
        fields_of_study: list[str] | None = None,
        inserted_before: str | None = None,
    ) -> DocumentCollection:
        """Create a document collection from dense retrieval."""
        documents: list[Document]
        match dataset.provider:
            case "vespa":
                documents = await fetch_from_vespa_dense_retrieval(
                    queries=queries,
                    search_iteration=search_iteration,
                    fields=[*BASIC_FIELDS, "snippets"],
                    top_k=top_k,
                    dataset=dataset,
                    time_range=time_range,
                    venues=venues,
                    authors=authors,
                    corpus_ids=corpus_ids,
                    fields_of_study=fields_of_study,
                    context=self._context,
                    inserted_before=inserted_before,
                )
        collection = self.from_docs(documents=documents)
        try:
            cache = self._context.cache
            await cache.put(collection.documents, collection.to_field_requirements(S2_FIELDS))
            collection = await collection.with_fields(["markdown"])
        except Exception as e:
            logging.exception(f"Failed to populate cache for dense retrieval documents (skipping): {e}")
        return collection
