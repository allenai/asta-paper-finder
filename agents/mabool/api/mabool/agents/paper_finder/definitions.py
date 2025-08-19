from __future__ import annotations

from ai2i.dcollection import CorpusId
from pydantic import Field

from mabool.data_model.agent import AgentInput, AgentOperationMode
from mabool.data_model.aliases import NaturalLanguageText


class PaperFinderInput(AgentInput):
    query: NaturalLanguageText
    anchor_corpus_ids: list[CorpusId] = Field(default_factory=list)
    operation_mode: AgentOperationMode = "infer"
