from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, TypedDict

from ai2i.dcollection import CorpusId, DocumentCollection, ExtractedYearlyTimeRange, RelevanceCriteria
from pydantic import BaseModel

from mabool.data_model.specifications import Specifications

type AgentOperationMode = Literal["infer", "fast", "diligent"]


class ExtractedContent(BaseModel):
    content: Optional[str]


class ExtractedAuthors(BaseModel):
    authors: list[str]


class ExtractedVenues(BaseModel):
    venues: list[str]


class ExtractedRecency(BaseModel):
    recency: Optional[Literal["first", "last"]]


class ExtractedCentrality(BaseModel):
    centrality: Optional[Literal["first", "last"]]


BroadOrSpecificLiterals = Literal["broad", "specific", "unknown"]


class BroadOrSpecificType(BaseModel):
    type: BroadOrSpecificLiterals


class ByNameOrTitleType(BaseModel):
    type: Literal["name", "title"]


class ExtractedFields(TypedDict):
    content: ExtractedContent
    authors: ExtractedAuthors
    venues: ExtractedVenues
    recency: ExtractedRecency
    centrality: ExtractedCentrality
    time_range: ExtractedYearlyTimeRange
    broad_or_specific: BroadOrSpecificType
    by_name_or_title: ByNameOrTitleType
    relevance_criteria: RelevanceCriteria
    domains: DomainsIdentified
    possible_refusal: PossibleRefusal


class MatchedTitle(BaseModel):
    matched_title: str | None
    matched_corpus_ids: list[CorpusId] | None


class ExtractedProperties(BaseModel):
    recent_first: bool = False
    recent_last: bool = False
    central_first: bool = False
    central_last: bool = False
    specific_paper_name: Optional[str] = None
    suitable_for_by_citing: Optional[bool] = None


class QueryType(BaseModel):
    type: Literal[
        "BROAD_BY_DESCRIPTION",
        "SPECIFIC_BY_TITLE",
        "SPECIFIC_BY_NAME",
        "BY_AUTHOR",
        "METADATA_ONLY_NO_AUTHOR",
        "UNKNOWN",
    ]
    broad_or_specific: BroadOrSpecificLiterals


# from https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_paper_relevance_search
FieldOfStudy = Literal[
    "Computer Science",
    "Mathematics",
    "Engineering",
    "Physics",
    "Materials Science",
    "Chemistry",
    "Biology",
    "Medicine",
    "Psychology",
    "Agricultural and Food Sciences",
    "Environmental Science",
    "Education",
    "Economics",
    "History",
    "Geology",
    "Art",
    "Geography",
    "Sociology",
    "Business",
    "Political Science",
    "Philosophy",
    "Law",
    "Linguistics",
    "Unknown",
]


class DomainsIdentified(BaseModel):
    main_field: FieldOfStudy
    key_secondary_fields: list[FieldOfStudy]


type RefusalType = Literal["similar to", "web access", "not paper finding", "affiliation", "author ID"]


class PossibleRefusal(BaseModel):
    type: RefusalType | None


class PartiallyAnalyzedQuery(BaseModel):
    original_query: str
    content: Optional[str]
    authors: list[str]
    venues: list[str]
    time_range: ExtractedYearlyTimeRange
    extracted_properties: ExtractedProperties
    query_type: QueryType | None
    relevance_criteria: RelevanceCriteria
    domains: DomainsIdentified | None = None
    possible_refusal: PossibleRefusal
    matched_title: MatchedTitle

    def to_analyzed_query(self) -> AnalyzedQuery:
        return AnalyzedQuery(
            original_query=self.original_query,
            content=self.content or "",
            authors=self.authors,
            venues=self.venues,
            time_range=self.time_range,
            extracted_properties=self.extracted_properties,
            query_type=self.query_type or QueryType(type="UNKNOWN", broad_or_specific="unknown"),
            relevance_criteria=self.relevance_criteria,
            domains=self.domains or DomainsIdentified(main_field="Unknown", key_secondary_fields=[]),
            possible_refusal=self.possible_refusal,
            matched_title=self.matched_title,
        )


class AnalyzedQuery(BaseModel):
    original_query: str
    content: str
    authors: list[str]
    venues: list[str]
    time_range: ExtractedYearlyTimeRange
    extracted_properties: ExtractedProperties
    query_type: QueryType
    relevance_criteria: RelevanceCriteria
    domains: DomainsIdentified
    possible_refusal: PossibleRefusal
    matched_title: MatchedTitle


class QueryAnalysisSuccess(BaseModel):
    analyzed_query: AnalyzedQuery
    specifications: Optional[Specifications] = None


class QueryAnalysisPartialSuccess(BaseModel):
    partially_analyzed_query: PartiallyAnalyzedQuery
    errors: list[QueryAnalyzerError]


class QueryAnalysisRefusal(BaseModel):
    analysis: AnalyzedQuery | PartiallyAnalyzedQuery
    errors: list[QueryAnalyzerError]


class QueryAnalysisFailure(BaseModel):
    partially_analyzed_query: PartiallyAnalyzedQuery
    error: NoActionableDataError


type QueryAnalysisResult = (
    QueryAnalysisSuccess | QueryAnalysisPartialSuccess | QueryAnalysisRefusal | QueryAnalysisFailure
)


def get_analyzed_query(query_analysis_result: QueryAnalysisResult) -> AnalyzedQuery | PartiallyAnalyzedQuery:
    match query_analysis_result:
        case QueryAnalysisSuccess(analyzed_query=analyzed_query):
            return analyzed_query
        case QueryAnalysisPartialSuccess(partially_analyzed_query=analyzed_query):
            return analyzed_query
        case QueryAnalysisRefusal(analysis=analyzed_query):
            return analyzed_query
        case QueryAnalysisFailure(partially_analyzed_query=analyzed_query):
            return analyzed_query


@dataclass
class QueryAnalyzerError(Exception):
    message: str


class ConflictingOptionsError(QueryAnalyzerError):
    def __init__(self, field: Literal["recent", "central"]) -> None:
        self.message = f"Query Analyzer failure: flagged both {field}_first and {field}_last"
        super().__init__(self.message)


class NoActionableDataError(QueryAnalyzerError):
    def __init__(self, message: Optional[str] = None) -> None:
        if message:
            self.message = message
        else:
            self.message = "Neither content nor authors were found in the query"
        super().__init__(self.message)


class AgentOutput(BaseModel):
    doc_collection: DocumentCollection
    response_text: str


class AgentInput(BaseModel):
    doc_collection: DocumentCollection


_ErrorTypeLiteral = Literal["query_refusal", "no_actionable_data_query", "heavy_load", "unable_to_do", "other"]
type ErrorType = _ErrorTypeLiteral


class AgentError(BaseModel):
    type: ErrorType
    message: str


class AggregatedMetrics(BaseModel):
    num_docs_judged: int
    num_relevance_judgement_calls: int
    total_cost: float | None = None
    total_tokens: int | None = None
    start_time: datetime | None = None
    duration: float | None = None
    relevance_judgement_failures: int | None = None


class ExplainedAgentOutput(AgentOutput):
    input_query: str
    analyzed_query: AnalyzedQuery | None
    metrics: AggregatedMetrics | None = None
    error: AgentError | None = None
