from typing import TypedDict

from ai2i.chain import define_prompt_llm_call
from pydantic import BaseModel

from mabool.agents.common.common import InputQuery

# -------------- #
# Suggest Papers #
# -------------- #


class SuggestPapersInput(InputQuery):
    extra_hints: str | None
    n_suggestions: int
    domain_description: str


class SuggestedPaper(BaseModel):
    title: str
    year: int | None


class SuggestedPapers(BaseModel):
    papers: list[SuggestedPaper]


_suggested_paper_prompt_tmpl = """
Suggest up to {n_suggestions} existing published research papers that match the following query:
Query: {query}

The requested papers are likely related to {domain_description}.
The paper titles you provide must be accurate as they will be used to search for the papers in a database.
{extra_hints}
"""  # noqa: E501

suggested_paper = define_prompt_llm_call(
    _suggested_paper_prompt_tmpl,
    input_type=SuggestPapersInput,
    output_type=SuggestedPapers,
).map(lambda o: o.papers)

# ------------------------ #
# Validate Suggested Paper #
# ------------------------ #


class SuggestedPaperDict(TypedDict):
    title: str
    year: int | None


class FoundPaperDict(TypedDict):
    title: str | None
    year: int | None
