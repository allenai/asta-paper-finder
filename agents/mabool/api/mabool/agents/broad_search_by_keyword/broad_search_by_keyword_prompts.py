from typing import TypedDict

from ai2i.chain import define_prompt_llm_call
from pydantic import BaseModel

# ------------ #
# Broad Search #
# ------------ #


_broad_search_prompt_tmpl = """
Given a user-provided natural language description of desired scientific papers, \
reformulate the query as an alternative search query for the Semantic Scholar search engine.
Focus on formulating the natural language query into a keyword search query, that is, \
remove unnecessary descriptive words that wont show up in the content itself and keep the keywords to look for.

Use plain text for queries, as Semantic Scholar does not support special syntax.
Semantic Scholar does not support hyphens in queries, so avoid hyphens.

When building the queries, try to use only content-keywords, that is, do emit metadata or non keyword-y wordings!

Input description: ```{paper_description}```
"""


class FormulateBroadSearchQueryInput(TypedDict):
    paper_description: str


class BroadSearchSuggestedQueries(BaseModel):
    keyword_query: str


broad_search = (
    define_prompt_llm_call(
        _broad_search_prompt_tmpl,
        input_type=FormulateBroadSearchQueryInput,
        output_type=BroadSearchSuggestedQueries,
        custom_format_instructions=(
            'Return a JSON dict with the key "keyword_query" and the value the query reformulated as a keyword query.'
        ),
    )
    .map(lambda o: o.keyword_query)
    .contra_map(lambda s: {"paper_description": s}, input_type=str)
)
