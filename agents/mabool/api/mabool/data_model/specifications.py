from __future__ import annotations

from typing import Any, ClassVar, Generic, Literal, Optional, Self, TypeVar

from ai2i.dcollection.interface.document import ExtractedYearlyTimeRange as Years
from pydantic import BaseModel, model_validator

fields_of_study = [
    "Computer Science",
    "Medicine",
    "Chemistry",
    "Biology",
    "Materials Science",
    "Physics",
    "Geology",
    "Psychology",
    "Art",
    "History",
    "Geography",
    "Sociology",
    "Business",
    "Political Science",
    "Economics",
    "Philosophy",
    "Mathematics",
    "Engineering",
    "Environmental Science",
    "Agricultural and Food Sciences",
    "Education",
    "Law",
    "Linguistics",
]
fields_of_study_set = set(map(str.lower, fields_of_study))


def _normalize(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return x.strip().lower()


class BaseSpec(BaseModel):
    def __repr_args__(self) -> list[tuple[str | None, Any]]:
        return [(k, v) for k, v in super().__repr_args__() if v is not None]

    def delete(self, *names: str) -> Self:
        return self.model_copy(update={name: None for name in names})

    def is_empty(self) -> bool:
        """Check if the object is empty (all fields are None or empty)."""
        for field in self.model_fields:
            value = getattr(self, field)
            match value:
                case str() as s:
                    if s.strip():
                        return False
                case list() as l if l:
                    return False
                case _:
                    if value is not None:
                        return False
        # If all fields are None or empty, return True
        return True
        # return all(getattr(self, field) is None for field in self.model_fields)


T = TypeVar("T")


class Set(BaseSpec, Generic[T]):
    op: Literal["and", "or"]
    items: list[T]

    @model_validator(mode="after")
    def more_than_one_item(self) -> Self:
        if len(self.items) < 2:
            raise ValueError("Set must contain at least two items")
        return self


type Composite[T] = T | Set[T] | None


class PaperSet(BaseSpec):
    op: Literal["any_author_of", "all_authors_of"]
    items: list[PaperSpec]


class AuthorSpec(BaseSpec):
    name: Optional[str] = None
    affiliation: Optional[str] = None
    papers: Optional[PaperSet] = None
    min_authors: Optional[int] = None

    dummies: ClassVar[set[str]] = {
        "john doe",
        "jane doe",
        "anonymous",
        "unknown",
        "example author",
        "example",
        "dummy author",
        "dummy",
    }

    @model_validator(mode="after")
    def name_is_not_empty(self) -> Self:
        if self.name is not None and not self.name.strip():
            raise ValueError("Author name cannot be empty")
        return self

    @model_validator(mode="after")
    def no_dummy_name(self) -> Self:
        if not self.name:
            return self
        name = self.name.strip().lower()
        if name in self.dummies:
            raise ValueError("Author name cannot be a dummy name")
        return self

    @model_validator(mode="after")
    def not_all_none(self) -> Self:
        if not any(getattr(self, field) is not None for field in self.model_fields if field != "min_authors"):
            raise ValueError("At least one field other than min_authors must be set")
        return self

    @model_validator(mode="after")
    def min_authors_use_any(self) -> Self:
        if self.min_authors is not None and self.papers and self.papers.op == "all_authors_of":
            raise ValueError("min_authors cannot be used with all_authors_of")
        return self


type References = Composite[PaperSpec]


type MinCitations = int | Literal["high"] | None


class PaperSpec(BaseSpec):
    name: Optional[str] = None
    full_name: Optional[str] = None
    field_of_study: Optional[str] = None
    content: Optional[str] = None
    years: Composite[Years] = None
    venue: Composite[str] = None
    venue_group: Composite[str] = None
    publication_type: Composite[str] = None
    min_citations: MinCitations = None
    authors: Composite[AuthorSpec] = None
    min_total_authors: Optional[int] = None
    citing: References = None
    cited_by: References = None
    exclude: Optional[PaperSpec] = None

    @model_validator(mode="after")
    def content_is_field_of_study(self) -> Self:
        if self.content and not self.field_of_study:
            content = _normalize(self.content)
            if content in fields_of_study_set:
                self.field_of_study = content
                self.content = None
        return self

    @model_validator(mode="after")
    def field_of_study_to_title_case(self) -> Self:
        if self.field_of_study:
            self.field_of_study = self.field_of_study.title()
        return self

    @model_validator(mode="after")
    def at_least_one_positve_field(self) -> Self:
        if not any(getattr(self, field) is not None for field in self.model_fields if field != "exclude"):
            raise ValueError("At least one field must be set")
        return self

    @model_validator(mode="after")
    def no_self_contradiction(self) -> Self:
        if self.exclude:
            for field in self.exclude.model_fields:
                pos_value = getattr(self, field)
                neg_value = getattr(self.exclude, field)
                if neg_value is not None and pos_value is not None and neg_value == pos_value:
                    raise ValueError(f"Field '{field}' cannot be both set and excluded with the same value")
        return self

    def normalize(self) -> PaperSpec:
        return self.model_copy(update={"name": _normalize(self.name), "content": _normalize(self.content)})

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PaperSpec):
            return False
        a = self.normalize()
        b = value.normalize()
        return super(PaperSpec, a).__eq__(b)

    def is_non_trivial_metadata_only(self) -> bool:
        """Check if the specification is non-trivial but metadata-only."""
        if self.is_empty():
            return False
        if self.content or self.name or self.full_name:
            return False
        author_is_trivial = False
        match self.authors:
            case AuthorSpec(name=str()):
                author_is_trivial = True
            case _:
                pass
        has_other_fields = any(
            getattr(self, field) is not None
            for field in self.model_fields
            if field not in {"content", "name", "full_name", "authors"}
        )
        if has_other_fields or not author_is_trivial:
            return True
        return False


class Specifications(BaseSpec):
    union: list[PaperSpec]
    error: Optional[str] = None

    def is_non_trivial_metadata_only(self) -> bool:
        """Check if the specifications are non-trivial but metadata-only."""
        match self.union:
            case [spec] if not spec.is_non_trivial_metadata_only():
                return False
            case _:
                pass
        return any(spec.is_non_trivial_metadata_only() for spec in self.union)
