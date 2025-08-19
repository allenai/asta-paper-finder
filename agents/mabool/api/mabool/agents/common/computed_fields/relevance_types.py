from typing import Any, Literal, TypedDict

from ai2i.dcollection import CorpusId
from pydantic import BaseModel


class RelevanceThresholds:
    NOT_RELEVANT = 0.25
    SOMEWHAT_RELEVANT = 0.67
    HIGHLY_RELEVANT = 0.99


class RelevanceLabels:
    PERFECTLY_RELEVANT = "Perfectly Relevant"
    HIGHLY_RELEVANT = "Highly Relevant"
    SOMEWHAT_RELEVANT = "Somewhat Relevant"
    NOT_RELEVANT = "Not Relevant"


RELEVANCE_LABEL_TO_SCORE = {
    RelevanceLabels.PERFECTLY_RELEVANT: 3,
    RelevanceLabels.HIGHLY_RELEVANT: 2,
    RelevanceLabels.SOMEWHAT_RELEVANT: 1,
    RelevanceLabels.NOT_RELEVANT: 0,
}


class LLMJudgementResult(TypedDict):
    doc_id: CorpusId
    model_name: str
    criteria_judgements: dict[str, Any]


class DocumentRelevanceInput(TypedDict):
    doc_id: CorpusId
    document: str
    criteria: str


class RelevanceScores(TypedDict):
    judgements_by_doc: dict[CorpusId, list[dict[str, Any]]]
    summaries_by_doc: dict[CorpusId, str | None]
    models_by_doc: dict[CorpusId, str]


class RelevanceCriterionJudgementValue(BaseModel):
    relevant_snippet: str | None
    relevance: Literal["Perfectly Relevant", "Somewhat Relevant", "Not Relevant"]


class RelevanceCriterionJudgementResult(RelevanceCriterionJudgementValue):
    name: str
