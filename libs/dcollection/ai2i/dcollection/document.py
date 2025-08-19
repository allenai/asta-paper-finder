from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence, overload

from pydantic import Field, PrivateAttr
from pydantic.fields import FieldInfo

from ai2i.common.utils.data_struct import SortedSet
from ai2i.dcollection.data_access_context import DynamicField
from ai2i.dcollection.fusers.fuse import fuse_origin_query, fuse_snippet
from ai2i.dcollection.interface.collection import (
    DocLoadingError,
    Document,
    dynamic_field,
    external_dynamic_field,
    fuse_citation_contexts,
)
from ai2i.dcollection.interface.document import (
    Author,
    Citation,
    CitationContext,
    CorpusId,
    DocumentFieldName,
    Journal,
    OriginQuery,
    PublicationVenue,
    RelevanceJudgement,
    Snippet,
)
from ai2i.dcollection.loaders.fields import load_markdown
from ai2i.dcollection.loaders.s2_rest import from_s2


class PaperFinderDocument(Document):
    @staticmethod
    def from_dict(params: dict[str, Any]) -> PaperFinderDocument:
        return PaperFinderDocument(**params)

    @staticmethod
    def get_predefined_dynamic_fields() -> dict[str, DynamicField]:
        static_fields = {
            field_name: field
            for field_name, field in PaperFinderDocument.model_fields.items()
            if isinstance(field, DynamicField)
        }
        return static_fields

    @staticmethod
    def get_static_fields() -> Mapping[str, FieldInfo]:
        return {
            field_name: field
            for field_name, field in PaperFinderDocument.model_fields.items()
            if not isinstance(field, DynamicField)
        }

    @property
    def entity_id(self) -> CorpusId:
        return self.corpus_id

    @entity_id.setter
    def entity_id(self, value: CorpusId) -> None:
        self.corpus_id = value

    corpus_id: CorpusId = Field()

    url: str | None = dynamic_field(default=None, loaders=[from_s2])
    title: str | None = dynamic_field(default=None, loaders=[from_s2])
    year: int | None = dynamic_field(default=None, loaders=[from_s2])
    authors: list[Author] | None = dynamic_field(default=None, loaders=[from_s2])
    abstract: str | None = dynamic_field(default=None, loaders=[from_s2])
    venue: str | None = dynamic_field(default=None, loaders=[from_s2])
    publication_venue: PublicationVenue | None = dynamic_field(default=None, loaders=[from_s2])
    publication_types: list[str] | None = dynamic_field(default=None, loaders=[from_s2])
    fields_of_study: list[str] | None = dynamic_field(default=None, loaders=[from_s2])
    tldr: str | None = dynamic_field(default=None, loaders=[from_s2])
    snippets: list[Snippet] | None = dynamic_field(default=None, fuse=fuse_snippet)
    origins: list[OriginQuery] | None = dynamic_field(default=None, fuse=fuse_origin_query)
    citations: list[Citation] | None = dynamic_field(default=None, loaders=[from_s2])
    citation_count: int | None = dynamic_field(default=None, loaders=[from_s2])
    influential_citation_count: int | None = dynamic_field(default=None, loaders=[from_s2])
    references: list[Citation] | None = dynamic_field(default=None, loaders=[from_s2])
    reference_count: int | None = dynamic_field(default=None, loaders=[from_s2])
    journal: Journal | None = dynamic_field(default=None, loaders=[from_s2])
    publication_date: date | None = dynamic_field(default=None, loaders=[from_s2])
    relevance_judgement: RelevanceJudgement | None = external_dynamic_field(default=None)
    markdown: str | None = dynamic_field(
        default=None,
        loaders=[load_markdown],
        required_fields=[
            "title",
            "authors",
            "year",
            "abstract",
            "snippets",
            "citation_contexts",
        ],
    )
    rerank_score: float | None = external_dynamic_field(default=None)
    final_agent_score: float | None = external_dynamic_field(default=None)
    citation_contexts: list[CitationContext] | None = dynamic_field(default=None, fuse=fuse_citation_contexts)

    _loaded_fields: set[DocumentFieldName] = PrivateAttr(default_factory=SortedSet)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        extra_keys = self.model_extra.keys() if self.model_extra else []
        for key, value in data.items():
            if key in extra_keys:
                self[key] = value
            else:
                self._add_to_loaded_fields(key)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PaperFinderDocument):
            return False
        return self.entity_id == other.entity_id

    def __hash__(self) -> int:
        return hash(self.entity_id)

    def __setattr__(self, key: str, value: Any) -> None:
        self._add_to_loaded_fields(key)
        super().__setattr__(key, value)
        field = self.fields().get(key)
        if self.model_extra is not None and (isinstance(field, DynamicField) and field.extra):
            self.model_extra[key] = value

    def fields(self) -> Mapping[str, DynamicField | FieldInfo]:
        return {
            **PaperFinderDocument.get_static_fields(),
            **PaperFinderDocument.get_predefined_dynamic_fields(),
            **self.dynamic_fields,
        }

    def _add_to_loaded_fields(self, key: str) -> None:
        if key != "dynamic_fields" and key not in self._loaded_fields:
            self._loaded_fields.add(key)

    def is_loaded(self, field_name: DocumentFieldName) -> bool:
        return field_name in self._loaded_fields

    def get_loaded_fields(self) -> set[DocumentFieldName]:
        return self._loaded_fields

    def clear_loaded_field(self, field_name: DocumentFieldName) -> None:
        self[field_name] = None
        if field_name in self._loaded_fields:
            self._loaded_fields.remove(field_name)

    @overload
    def dynamic_value[V](
        self,
        field_name: DocumentFieldName,
        field_type: type[V],
        /,
        default: None = None,
    ) -> V | None: ...

    @overload
    def dynamic_value[V](self, field_name: DocumentFieldName, field_type: type[V], /, default: V) -> V: ...

    def dynamic_value[V](
        self,
        field_name: DocumentFieldName,
        field_type: type[V],
        /,
        default: V | None = None,
    ) -> V | None:
        return getattr(self, field_name, default)

    def __getitem__(self, field_name: DocumentFieldName) -> Any:
        return getattr(self, field_name, None)

    def __setitem__(self, field_name: DocumentFieldName, value: Any) -> None:
        setattr(self, field_name, value)

    def fuse(self, *entities: Document) -> Document:
        if not entities:
            return self
        for other in entities:
            if self.entity_id != other.entity_id:
                raise ValueError("Entity ids must match in order to fuse them.")
            for field_name, field in self.fields().items() | other.fields().items():
                if isinstance(field, DynamicField):
                    field.fuse(self, other, field_name)
        return self

    def clone_partial(self, fields: list[DocumentFieldName] | None = None) -> PaperFinderDocument:
        loaded_e = {
            field: self[field]
            for field in SortedSet(self.model_fields_set | (self.model_extra.keys() if self.model_extra else set()))
            if field == "dynamic_fields"
            or (
                self.is_loaded(field)
                and (
                    fields is None
                    or field in fields
                    or ((field_info := self.fields().get(field)) is not None and field_info.is_required())
                )
            )
        }
        return PaperFinderDocument(**loaded_e)

    def clone_with(self, data_overrides: dict[DocumentFieldName, Any]) -> PaperFinderDocument:
        loaded_e = {
            field: self[field]
            for field in SortedSet(self.model_fields_set | (self.model_extra.keys() if self.model_extra else set()))
            if self.is_loaded(field) or field == "dynamic_fields"
        }
        for k, v in data_overrides.items():
            if len(getattr(self.fields()[k], "loading_functions", [])) > 0:
                raise ValueError(f"Cannot replace field {k} as it is has a loader defined.")
            loaded_e[k] = v
        return PaperFinderDocument(**loaded_e)

    def assign_loaded_values(
        self,
        field_names: Sequence[DocumentFieldName],
        loaded_entities: Sequence[Document],
    ) -> None:
        for loaded_entity in loaded_entities:
            for field_name in field_names:
                attr = getattr(loaded_entity, field_name)
                if isinstance(attr, DocLoadingError):
                    self.clear_loaded_field(field_name)
                else:
                    setattr(self, field_name, attr)

    def __repr__(self) -> str:
        # this is used for logging where we log the first document in a collection
        return f"Document(corpus_id={self.corpus_id}, title={self.title}, year={self.year})"
