import logging
from typing import TypedDict

from ai2i.chain import Timeouts, define_prompt_llm_call
from ai2i.dcollection import DocumentCollection
from pydantic import BaseModel

from mabool.agents.query_analyzer.query_analyzer import get_default_endpoint

logger = logging.getLogger(__name__)


class CombineAnchorInput(TypedDict):
    query: str
    anchors_markdown: str


class CombineAnchorOutput(BaseModel):
    combined_query: str


_combined_anchor_query_prompt_tmpl = """
Given the below query, and a list of anchor documents, combine the query with the anchor documents to form a new query.

The new query should be mainly based on the original query, \
but could include additional clarification from the anchor documents.

This clarifications should help, whenever necessary, disambiguate the query, \
or provide context for potentially under-specified terms. \
The additions should be kept minimal and not change the main intent of the original query.

Original Query: {{query}}

Anchor Documents:
{{anchors_markdown}}
"""

combined_anchor_query = define_prompt_llm_call(
    _combined_anchor_query_prompt_tmpl,
    format="mustache",
    input_type=CombineAnchorInput,
    output_type=CombineAnchorOutput,
)


async def combine_content_query_with_anchors(content_query: str, anchor_docs: DocumentCollection) -> str:
    if len(anchor_docs) == 0:
        return content_query
    try:
        anchor_docs_markdown = "\n".join(anchor_docs.project(lambda doc: doc.markdown or ""))
        endpoint = get_default_endpoint().timeout(Timeouts.medium)
        anchor_combined_query = await endpoint.execute(combined_anchor_query).once(
            {"query": content_query, "anchors_markdown": anchor_docs_markdown}
        )
        return anchor_combined_query.combined_query
    except Exception as e:
        logger.exception(f"Failed to combine content query with anchor documents: {content_query}. Error: {e}")
        return content_query
