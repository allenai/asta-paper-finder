from __future__ import annotations

from datetime import date
from functools import total_ordering
from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field
from semanticscholar.Journal import Journal as S2Journal
from semanticscholar.PublicationVenue import PublicationVenue as S2PublicationVenue

type CorpusId = str

type Year = int


class ExtractedYearlyTimeRange(BaseModel):
    start: Year | None = None
    end: Year | None = None

    def is_empty(self) -> bool:
        return self.start is None and self.end is None

    def non_empty(self) -> bool:
        return not self.is_empty()

    def __hash__(self) -> int:
        return hash((self.start, self.end))


class Citation(BaseModel):
    target_corpus_id: int
    reference_count: int | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    is_influential: bool | None = None
    num_contexts: int | None = None
    year: int | None = None
    publication_date: date | None = None


class CitationContext(BaseModel):
    text: str
    source_corpus_id: CorpusId | None = None
    within_snippet_offset_start: int | None = None
    within_snippet_offset_end: int | None = None
    similarity_score: float | None = None

    model_config = ConfigDict(frozen=True)

    def mark_within_snippet_offset(self, before: str = "<<<", after: str = ">>>") -> str:
        if self.within_snippet_offset_start is not None and self.within_snippet_offset_end is not None:
            return (
                self.text[: self.within_snippet_offset_start]
                + before
                + self.text[self.within_snippet_offset_start : self.within_snippet_offset_end]
                + after
                + self.text[self.within_snippet_offset_end :]
            )
        return self.text


class PublicationVenue(BaseModel):
    normalized_name: str
    alternate_names: list[str] | None = None

    @classmethod
    def from_dict(cls, s2_publication_venue: S2PublicationVenue) -> PublicationVenue:
        return PublicationVenue(
            normalized_name=s2_publication_venue.name,
            alternate_names=s2_publication_venue.alternate_names,
        )


class Journal(BaseModel):
    name: str | None = None
    # NOTE: defined differently than in SemanticScholar package as we've seen proof for str volumes
    volume: str | None = None
    pages: str | None = None

    @classmethod
    def from_dict(cls, journal: S2Journal) -> Journal:
        return Journal(
            name=journal.name,
            volume=str(journal.volume) if journal.volume else None,
            pages=journal.pages,
        )


class Author(BaseModel):
    name: str
    author_id: str | None = None

    def __str__(self) -> str:
        return self.name


@total_ordering
class RelevanceJudgement(BaseModel):
    relevance: int = Field(..., ge=-1, le=3)
    relevance_model_name: str | None = None
    relevance_criteria_judgements: list[RelevanceCriterionJudgement] | None = None
    relevance_score: float | None = None
    relevance_summary: str | None = None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RelevanceJudgement):
            return NotImplemented
        return self.relevance == other.relevance

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, RelevanceJudgement):
            return NotImplemented
        return self.relevance < other.relevance

    def __float__(self) -> float:
        return float(self.relevance)


@total_ordering
class SimilarityScore(BaseModel):
    similarity_model_name: str
    query: str
    score: float

    def __hash__(self) -> int:
        return hash((self.similarity_model_name, self.query, self.score))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SimilarityScore):
            return NotImplemented
        return (
            self.score == other.score
            and self.query == other.query
            and self.similarity_model_name == other.similarity_model_name
        )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, SimilarityScore):
            return NotImplemented
        return self.score < other.score

    def __float__(self) -> float:
        return self.score


class BoundingBox(BaseModel):
    page: int
    top: float
    left: float
    h: float
    w: float


section_kind_order = {
    "title": 0,
    "abstract": 1,
    "body": 2,
    None: 3,
}


class RefMention(BaseModel):
    matched_paper_corpus_id: str
    within_snippet_offset_start: int | None = None
    within_snippet_offset_end: int | None = None


class Offset(BaseModel):
    start: int
    end: int


class SentenceOffsets(BaseModel):
    global_offset: Offset | None = None
    within_snippet_offset: Offset | None = None


class HasIsLoaded(Protocol):
    def is_loaded(self, e: str) -> bool: ...


@total_ordering
class Snippet(BaseModel):
    text: str
    section_title: str | None = None
    section_kind: Literal["title", "abstract", "body"] | None = None
    ref_mentions: list[RefMention] | None = None
    char_start_offset: int | None = None
    char_end_offset: int | None = None
    similarity_scores: list[SimilarityScore] | None = None
    bounding_boxes: list[BoundingBox] | None = None
    sentences: list[SentenceOffsets] | None = None

    class Config:
        validate_assignment = True

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Snippet) and self.text == other.text and self.char_start_offset == other.char_start_offset
        )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Snippet):
            return NotImplemented
        return (
            section_kind_order.get(self.section_kind, max(section_kind_order.values()) + 1),
            self.char_start_offset if self.char_start_offset is not None else 0,
            self.char_end_offset if self.char_end_offset is not None else 0,
            self.text,
        ) < (
            section_kind_order.get(other.section_kind, max(section_kind_order.values()) + 1),
            other.char_start_offset if other.char_start_offset is not None else 0,
            other.char_end_offset if other.char_end_offset is not None else 0,
            other.text,
        )

    def __hash__(self) -> int:
        return hash((self.text, self.char_start_offset))


@total_ordering
class Sentence(BaseModel):
    text: str
    section_title: str | None = None
    section_kind: Literal["title", "abstract", "body"] | None = None
    cited_s2_ids: list[int] | None = None
    char_start_offset: int | None = None
    char_end_offset: int | None = None
    similarity_scores: list[SimilarityScore] | None = None
    bounding_boxes: list[BoundingBox] | None = None

    class Config:
        validate_assignment = True

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Sentence)
            and self.text == other.text
            and self.char_start_offset == other.char_start_offset
        )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Sentence):
            return NotImplemented
        return (
            section_kind_order.get(self.section_kind, max(section_kind_order.values()) + 1),
            self.char_start_offset if self.char_start_offset else 0,
            self.text,
        ) < (
            section_kind_order.get(other.section_kind, max(section_kind_order.values()) + 1),
            other.char_start_offset if other.char_start_offset else 0,
            other.text,
        )

    def __hash__(self) -> int:
        return hash((self.text, self.char_start_offset))

    @staticmethod
    def from_snippet(snippet: Snippet) -> Sentence:
        return Sentence(
            text=snippet.text,
            section_title=snippet.section_title,
            section_kind=snippet.section_kind,
            char_start_offset=snippet.char_start_offset,
            char_end_offset=snippet.char_end_offset,
            cited_s2_ids=(
                [int(ref.matched_paper_corpus_id) for ref in snippet.ref_mentions] if snippet.ref_mentions else []
            ),
            similarity_scores=snippet.similarity_scores,
            bounding_boxes=snippet.bounding_boxes,
        )


class S2PaperRelevanceSearchQuery(BaseModel):
    query: str
    num_results: int
    time_range: ExtractedYearlyTimeRange | None = None
    venues: Sequence[str] | None = None

    def __hash__(self) -> int:
        return hash((self.query, self.num_results, self.time_range, ",".join(self.venues or [])))


class S2PaperTitleSearchQuery(BaseModel):
    query: str
    time_range: ExtractedYearlyTimeRange | None = None
    venues: Sequence[str] | None = None

    def __hash__(self) -> int:
        return hash((self.query, self.time_range, ",".join(self.venues or [])))


class S2AuthorPaperSearchQuery(BaseModel):
    author_ids: list[str]

    def __hash__(self) -> int:
        return hash(",".join(self.author_ids))


class S2CitingPapersQuery(BaseModel):
    corpus_id: CorpusId

    def __hash__(self) -> int:
        return hash(self.corpus_id)


S2PaperOriginQuery = (
    S2PaperRelevanceSearchQuery | S2PaperTitleSearchQuery | S2AuthorPaperSearchQuery | S2CitingPapersQuery
)


class OriginQuery(BaseModel):
    query_type: Literal[
        "dense",
        "s2_relevance_search",
        "s2_bulk_search",
        "s2_title_search",
        "s2_author_paper_search",
        "s2_citing_papers",
        "snowball",
        "llm",
    ]
    provider: str | None = None
    dataset: str | None = None
    variant: str | None = None
    query: str | S2PaperOriginQuery
    iteration: int | None = None
    ranks: list[int] | None = None

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, OriginQuery)
            and self.query_type == other.query_type
            and self.provider == other.provider
            and self.dataset == other.dataset
            and self.variant == other.variant
            and self.query == other.query
            and self.iteration == other.iteration
        )

    def __hash__(self) -> int:
        return hash((self.query_type, self.provider, self.dataset, self.variant, self.query, self.iteration))

    def __repr__(self) -> str:
        return " | ".join(
            filter(None, [str(self.query_type), self.provider, self.dataset, self.variant, str(self.query)])
        )


DEFAULT_CONTENT_RELEVANCE_CRITERION_NAME = "Relevance Criterion"


class RelevanceCriterion(BaseModel):
    name: str
    description: str
    weight: float

    model_config = ConfigDict(frozen=True)


class RelevanceCriteria(BaseModel):
    query: str
    required_relevance_critieria: list[RelevanceCriterion] | None = None
    nice_to_have_relevance_criteria: list[RelevanceCriterion] | None = None
    clarification_questions: list[str] | None = None

    model_config = ConfigDict(frozen=True)

    def to_flat_criteria(self, include_nice_to_have: bool = True) -> list[RelevanceCriterion]:
        return (self.required_relevance_critieria if self.required_relevance_critieria else []) + (
            self.nice_to_have_relevance_criteria
            if include_nice_to_have and self.nice_to_have_relevance_criteria
            else []
        )

    def is_default(self) -> bool:
        return (
            self.required_relevance_critieria is not None
            and len(self.required_relevance_critieria) == 1
            and self.required_relevance_critieria[0].name == DEFAULT_CONTENT_RELEVANCE_CRITERION_NAME
        )

    @staticmethod
    def to_default_content_criteria(relevance_criteria: RelevanceCriteria, content: str) -> RelevanceCriteria:
        return RelevanceCriteria(
            **relevance_criteria.model_dump(exclude={"required_relevance_critieria"}),
            required_relevance_critieria=[
                RelevanceCriterion(name=DEFAULT_CONTENT_RELEVANCE_CRITERION_NAME, description=content, weight=1.0)
            ],
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RelevanceCriteria):
            return False
        return bool(
            self.query == other.query
            and self.required_relevance_critieria == other.required_relevance_critieria
            and self.nice_to_have_relevance_criteria == other.nice_to_have_relevance_criteria
            and self.clarification_questions == other.clarification_questions
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.query,
                tuple(self.required_relevance_critieria or []),
                tuple(self.nice_to_have_relevance_criteria or []),
                tuple(self.clarification_questions or []),
            )
        )


class RelevanceCriterionJudgement(BaseModel):
    name: str
    relevance: int = Field(ge=0, le=3)
    relevant_snippets: list[Snippet | CitationContext] | None = None


type SortOrder = Literal["asc", "desc"]

rj_4l_codes = {
    "Perfectly Relevant": 3,
    "Highly Relevant": 2,
    "Somewhat Relevant": 1,
    "Not Relevant": 0,
}


type SampleMethod = Literal[
    "random_stratified_relevance", "random", "top_relevance", "bottom_origin_rank_stratified_relevance"
]
DocumentFieldName = (
    Literal[
        "corpus_id",
        "url",
        "title",
        "year",
        "authors",
        "abstract",
        "venue",
        "publication_venue",
        "publication_types",
        "fields_of_study",
        "tldr",
        "snippets",
        "relevance_judgement",
        "origins",
        "markdown",
        "citations",
        "references",
        "citation_count",
        "reference_count",
        "influential_citation_count",
        "rerank_score",
        "final_agent_score",
        "citation_contexts",
        "relevance_criteria",
        "publication_date",
        "journal",
    ]
    | str
)
