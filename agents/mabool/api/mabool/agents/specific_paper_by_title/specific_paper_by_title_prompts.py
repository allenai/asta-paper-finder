from ai2i.chain import define_prompt_llm_call
from pydantic import BaseModel

from mabool.agents.common.common import InputQuery, as_input_query

# ------------- #
# Extract Title #
# ------------- #

_title_extraction_prompt_tmpl = """
Given a paper finding query, extract the paper title from the query:

Query: ```{query}```
"""  # noqa: E501


class ExtractTitlePromptOutput(BaseModel):
    title: str | None


title_extraction = (
    define_prompt_llm_call(
        _title_extraction_prompt_tmpl,
        input_type=InputQuery,
        output_type=ExtractTitlePromptOutput,
        custom_format_instructions=(
            'Return a JSON dict with the key "title" and the value is a string representing the extracted title,'
            + " if found or null otherwise."
        ),
    )
    .map(lambda o: o.title)
    .contra_map(as_input_query)
)
