from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from functools import partial
from typing import Any, Awaitable, Callable, Literal, Mapping, Protocol, Self, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo

from ai2i.dcollection.data_access_context import (
    ComputationId,
    DocumentCollectionContext,
    DocumentFieldLoader,
    DynamicallyLoadedEntity,
    DynamicField,
    EntityId,
    FieldRequirements,
    Fuser,
    SubsetCacheInterface,
)
from ai2i.dcollection.interface.document import (
    Author,
    Citation,
    CitationContext,
    CorpusId,
    DocumentFieldName,
    ExtractedYearlyTimeRange,
    Journal,
    OriginQuery,
    PublicationVenue,
    RelevanceJudgement,
    SampleMethod,
    Snippet,
    SortOrder,
)

Arm = int | float | str
Num = int | float


def fuse_citation_contexts[DFN: str](
    fuse_to: DynamicallyLoadedEntity[DFN],
    fuse_from: DynamicallyLoadedEntity[DFN],
    field: str,
) -> None:
    fuse_to_citation_contexts = getattr(fuse_to, field) if fuse_to.is_loaded(field) else []
    fuse_from_citation_contexts = getattr(fuse_from, field) if fuse_from.is_loaded(field) else []

    if fuse_to_citation_contexts or fuse_from_citation_contexts:
        fuse_to_citation_contexts_set = set(fuse_to_citation_contexts)
        # keep the order of the contexts
        # NOTE: this is sub-optimal, contexts should be sorted by relevance/similarity
        #       to the query, but we don't have this information here
        fused_citation_contexts = fuse_to_citation_contexts + [
            context for context in fuse_from_citation_contexts if context not in fuse_to_citation_contexts_set
        ]
        setattr(fuse_to, field, fused_citation_contexts)


class TakeFirst[DFN: str](Fuser[DFN]):
    def __call__(
        self,
        fuse_to: DynamicallyLoadedEntity[DFN],
        fuse_from: DynamicallyLoadedEntity[DFN],
        field: DFN,
    ) -> None:
        if fuse_to.is_loaded(field):
            return
        elif fuse_from.is_loaded(field):
            setattr(fuse_to, field, getattr(fuse_from, field, None))
        else:
            # Not loaded for either, so no need to do anything.
            pass


def dynamic_field(
    *args: Any,
    loaders: Sequence[DocumentFieldLoader] | None = None,
    fuse: Fuser[DocumentFieldName] = TakeFirst(),
    required_fields: Sequence[DocumentFieldName] | None = None,
    **kwargs: Any,
) -> Any:
    return DynamicField(*args, loaders=loaders, fuse=fuse, required_fields=required_fields, **kwargs)


def external_dynamic_field(
    *args: Any,
    fuse: Fuser[DocumentFieldName] = TakeFirst(),
    **kwargs: Any,
) -> Any:
    return DynamicField(*args, fuse=fuse, extra=True, mandatory_loader=True, **kwargs)


class DocLoadingError(Exception):
    def __init__(self, corpus_id: CorpusId, original_exception: Exception) -> None:
        self.corpus_id = corpus_id
        self.original_exception = original_exception
        super().__init__(f"Error in corpus {corpus_id}: {str(original_exception)}")

    def __str__(self):
        return f"CorpusError(corpus_id={self.corpus_id}): {str(self.original_exception)}"

    @property
    def entity_id(self) -> EntityId:
        return self.corpus_id


class Document(ABC, DynamicallyLoadedEntity[DocumentFieldName]):
    model_config = ConfigDict(extra="allow")

    corpus_id: CorpusId
    url: str | None
    title: str | None
    year: int | None
    authors: list[Author] | None
    abstract: str | None
    venue: str | None
    publication_venue: PublicationVenue | None
    publication_types: list[str] | None
    fields_of_study: list[str] | None
    tldr: str | None
    snippets: list[Snippet] | None
    origins: list[OriginQuery] | None
    citations: list[Citation] | None
    citation_count: int | None
    influential_citation_count: int | None
    references: list[Citation] | None
    reference_count: int | None
    relevance_judgement: RelevanceJudgement | None
    markdown: str | None
    rerank_score: float | None
    final_agent_score: float | None
    citation_contexts: list[CitationContext] | None
    journal: Journal | None
    publication_date: date | None

    @abstractmethod
    def get_loaded_fields(self) -> set[str]: ...

    @property
    @abstractmethod
    def entity_id(self) -> str: ...

    @abstractmethod
    def clone_with(self, data_overrides: dict[Any, Any]) -> Self: ...

    @abstractmethod
    def fuse(self, *entities: Document) -> Document: ...

    @abstractmethod
    def __getitem__(self, key: str) -> Any: ...

    @abstractmethod
    def __setitem__(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def clone_partial(self, fields: list[DocumentFieldName] | None = None) -> Self: ...

    @abstractmethod
    def assign_loaded_values(
        self, field_names: Sequence[DocumentFieldName], loaded_entities: Sequence[Document]
    ) -> None: ...

    @abstractmethod
    def dynamic_value[V](
        self, field_name: DocumentFieldName, field_type: type[V], /, default: V | None = None
    ) -> V | None: ...

    @abstractmethod
    def fields(self) -> Mapping[str, DynamicField | FieldInfo]: ...


class BaseDocumentCollectionFactory[G, V, B](BaseModel, ABC):
    @abstractmethod
    def cache(self) -> SubsetCacheInterface:
        pass

    @abstractmethod
    def context(self) -> DocumentCollectionContext:
        pass

    @abstractmethod
    def from_ids(self, corpus_ids: list[CorpusId]) -> DocumentCollection:
        """Create a document collection from a list of corpus IDs."""
        pass

    @abstractmethod
    def from_docs(
        self,
        documents: Sequence[Document],
        computed_fields: dict[DocumentFieldName, Any] | None = None,
    ) -> DocumentCollection:
        """Create a document collection from a list of documents."""
        pass

    @abstractmethod
    def empty(self) -> DocumentCollection:
        """Create an empty document collection."""
        pass

    @abstractmethod
    async def from_s2_by_author(
        self, authors_profiles: list[list[Any]], limit: int, inserted_before: str | None
    ) -> DocumentCollection:
        """Create a document collection from S2 by author."""
        pass

    @abstractmethod
    async def from_s2_by_title(
        self,
        query: str,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        inserted_before: str | None = None,
    ) -> DocumentCollection:
        """Create a document collection from S2 by title."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def from_s2_citing_papers(
        self,
        corpus_id: CorpusId,
        search_iteration: int = 1,
        total_limit: int = 1000,
        inserted_before: str | None = None,
    ) -> DocumentCollection:
        """Create a document collection from S2 citing papers."""
        pass

    @abstractmethod
    async def from_dense_retrieval(
        self,
        queries: list[str],
        search_iteration: int,
        dataset: Any,
        top_k: int,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        authors: list[str] | None = None,
        corpus_ids: list[CorpusId] | None = None,
        fields_of_study: list[str] | None = None,
        inserted_before: str | None = None,
    ) -> DocumentCollection:
        """Create a document collection from dense retrieval."""
        pass


class DocumentCollection(BaseModel, ABC):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    documents: Sequence[Document] = Field(default_factory=list)
    computed_fields: dict[DocumentFieldName, DynamicField] = Field(default_factory=dict, exclude=True)
    factory: BaseDocumentCollectionFactory = Field(exclude=True)

    @abstractmethod
    def map(self, map_fn: DocumentProjector[Document]) -> DocumentCollection: ...

    @abstractmethod
    def map_enumerate(self, map_fn: DocumentEnumProjector[Document]) -> DocumentCollection: ...

    @abstractmethod
    def filter(self, filter_fn: DocumentPredicate) -> DocumentCollection: ...

    @abstractmethod
    def merged(self, *collections: DocumentCollection) -> DocumentCollection: ...

    @abstractmethod
    def project[V](self, map_fn: DocumentProjector[V]) -> list[V]: ...

    @abstractmethod
    def group_by[V](self, group_fn: DocumentProjector[V]) -> dict[V, DocumentCollection]: ...

    @abstractmethod
    def take(self, n: int) -> DocumentCollection: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    async def with_fields(
        self, fields: Sequence[DocumentFieldName | BaseComputedField[DocumentFieldName, Any]]
    ) -> DocumentCollection: ...

    @abstractmethod
    def update_computed_fields(self, fields: Sequence[DocumentFieldName | BaseComputedField]) -> DocumentCollection: ...

    @abstractmethod
    def sorted(self, sort_definitions: Sequence[DocumentCollectionSortDef]) -> DocumentCollection: ...

    @abstractmethod
    def __add__(self, other: DocumentCollection) -> DocumentCollection: ...

    @abstractmethod
    def __sub__(self, other: DocumentCollection) -> DocumentCollection: ...

    @abstractmethod
    def subtract(self, other: DocumentCollection) -> DocumentCollection: ...

    @abstractmethod
    def multi_group_by[V](self, group_fn: DocumentProjector[Sequence[V]]) -> dict[V, DocumentCollection]: ...

    @abstractmethod
    def to_dataframe(
        self,
        fields: list[str],
        handle_missing_fields: Literal["raise", "fill", "skip_doc"] = "raise",
    ) -> pd.DataFrame: ...

    @abstractmethod
    def sample(self, n: int, method: SampleMethod) -> DocumentCollection: ...

    @abstractmethod
    def to_debug_dataframe(
        self,
    ) -> pd.DataFrame: ...

    @abstractmethod
    def to_field_requirements(
        self, field_names: Sequence[DocumentFieldName]
    ) -> Sequence[FieldRequirements[DocumentFieldName]]: ...


class QueryFn[DFN: str](Protocol):
    def __call__(
        self,
        entities: Sequence[DynamicallyLoadedEntity[DFN]],
        fields: list[DFN],
        context: DocumentCollectionContext,
    ) -> Awaitable[list[DynamicallyLoadedEntity[DFN]]]: ...


class DocumentPredicate(Protocol):
    def __call__(self, doc: Document) -> bool: ...


class DocumentProjector[V](Protocol):
    def __call__(self, doc: Document) -> V: ...


class DocumentEnumProjector[V](Protocol):
    def __call__(self, i: int, doc: Document) -> V: ...


class BaseComputedField[DFN: str, V](BaseModel):
    """
    Base class for computed fields in documents.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    field_name: DFN = Field()
    required_fields: list[DFN] = Field(default_factory=list)

    @property
    @abstractmethod
    def use_cache(self) -> bool:
        pass

    @property
    @abstractmethod
    def computation(self) -> Callable[[Any], V]:
        pass

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseComputedField):
            return False
        return bool(
            self.field_name == other.field_name
            and self.required_fields == other.required_fields
            and self.use_cache == other.use_cache
            and self.computation_id == other.computation_id
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.field_name,
                tuple(self.required_fields),
                self.use_cache,
                self.computation_id,
            )
        )

    @property
    def computation_id(self) -> ComputationId:
        return (
            (
                str(self.computation.func.__code__),
                self.computation.args,
                frozenset(self.computation.keywords.items()) if self.computation.keywords else None,
            )
            if isinstance(self.computation, partial)
            else str(self.computation.__code__)
        )


class DocumentCollectionSortDef(BaseModel):
    field_name: DocumentFieldName = Field()
    order: SortOrder = Field(default="asc")


BASIC_FIELDS: list[DocumentFieldName] = [
    "corpus_id",
    "url",
    "title",
    "year",
    "authors",
    "abstract",
    "venue",
]
UI_REQUIRED_FIELDS: list[DocumentFieldName] = ["publication_date", "journal", "citation_count"]
CITATION_FIELDS: list[DocumentFieldName] = [
    "citations",
    "references",
    "citation_count",
    "reference_count",
    "influential_citation_count",
]
S2_FIELDS: list[DocumentFieldName] = [
    *BASIC_FIELDS,
    *CITATION_FIELDS,
    "tldr",
]
ALL_FIELDS: list[DocumentFieldName] = [
    *BASIC_FIELDS,
    *CITATION_FIELDS,
    "tldr",
    "snippets",
    "relevance_judgement",
    "origins",
    "markdown",
    "rerank_score",
    "final_agent_score",
    "citation_contexts",
    "relevance_criteria",
]
