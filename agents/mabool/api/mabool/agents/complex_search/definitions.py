from ai2i.dcollection import DocumentCollection, ExtractedYearlyTimeRange, PaperFinderDocumentCollection
from pydantic import BaseModel, Field, field_validator

from mabool.agents.llm_suggestion.llm_suggestion_agent import LLMSuggestionArgs
from mabool.data_model.agent import AgentInput, DomainsIdentified, RelevanceCriteria
from mabool.utils.dc import DC


class BroadSearchArgs(LLMSuggestionArgs, BaseModel):
    content_query: str
    relevance_criteria: RelevanceCriteria
    recent_first: bool
    recent_last: bool
    central_first: bool
    central_last: bool
    time_range: ExtractedYearlyTimeRange | None = None
    venues: list[str] | None = None
    authors: list[str] | None = None
    domains: DomainsIdentified
    extracted_name: str | None = None
    suitable_for_by_citing: bool | None = None
    anchor_doc_collection: DocumentCollection = Field(default_factory=DC.empty)


class BroadSearchInput(BroadSearchArgs, AgentInput):
    content_query: str
    relevance_criteria: RelevanceCriteria
    recent_first: bool
    recent_last: bool
    central_first: bool
    central_last: bool
    time_range: ExtractedYearlyTimeRange | None = None
    venues: list[str] | None = None
    authors: list[str] | None = None
    domains: DomainsIdentified
    apply_relevance_judgement: bool = True
    extracted_name: str | None = None
    suitable_for_by_citing: bool | None = None
    anchor_doc_collection: DocumentCollection = Field(default_factory=DC.empty)


class BroadSearchOutput(BaseModel):
    doc_collection: DocumentCollection

    @field_validator("doc_collection", mode="after")
    @classmethod
    def copy_doc_collection(cls, doc_collection: PaperFinderDocumentCollection) -> DocumentCollection:
        copied_documents = [d.model_copy(deep=True) for d in doc_collection.documents]
        return doc_collection.factory.from_docs(copied_documents)
