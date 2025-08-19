from typing import TypedDict

from ai2i.chain import (
    define_chat_llm_call,
    define_prompt_llm_call,
    system_message,
    user_message,
)
from pydantic import BaseModel

# ------ #
# Common #
# ------ #


class DenseQueries(BaseModel):
    alternative_queries: list[str]


# --------------------- #
# Formulate Dense Query #
# --------------------- #

_dense_formulation_prompt_multiple_tmpl = """
Given the following search query, formulate {max_output} different natural language queries to run on a dense retrieval index to find papers that match the original search query.
The dense retrieval index contains research papers taken from the arXiv and ACL Anthology.
Be creative, and try to formulate queries that are different from each other.
Each passage in the dense index is a single sentence.
Drop phrases like "a paper about..." or "studies showing...".
- For example for the original search query "papers about efficient language modeling", a good query could be "methods for efficient language modeling", but a bad query would be "papers that talk about efficient language modeling".
DO NOT include general preferences such as "recent paper", "highly-cited paper", as these are not expected to be found in the text of the paper.
The index does not support logical operators like "AND", "OR", "NOT", "-", "+", "&" etc.

Original Search Query: ```{search_query}```
"""  # noqa: E501


class FormulateDenseQueriesInput(TypedDict):
    search_query: str
    max_output: int


dense_formulate = define_prompt_llm_call(
    _dense_formulation_prompt_multiple_tmpl, input_type=FormulateDenseQueriesInput, output_type=DenseQueries
).map(lambda o: o.alternative_queries)

_alternative_dense_formulation_prompt_tmpl = """
Your task is to come up with up to {{max_output}} alternative search queries that will help find passages that answer the following search query.
The queries will be run on a dense index that contains passages from academic research papers.
I am NOT looking for simple synonym paraphrases of common words, as these are captured by the index embeddings. Try using some reasoning to come up with interesting new ways to answer the original query.
Make sure you use wording that is actually used within the searched for domain. Don't just give arbitrary synonyms.
Drop phrases like "study about...", "research showing...", "evidence for...", as these are already true for all passages in the index of academic papers.

## Examples:
Example: "wide transformer models":
BAD output: "expansive transformer models"
- Reason: "expansive" is a bad synonym in this case, as it is not used to describe neural network architectures.
GOOD output: "shallow transformer models"
- Reason: the word shallow is commonly used to describe neural architectures that are wide but not deep.
GOOD output: "wide attention-based models"
- Reason: transformer are the most popular version of attention-based models, thus papers that talk about wide attention-based models are likely to be relevant for the original query

Output a json with a single key "alternative_queries" with its value a list of query strings.
"""  # noqa: E501

dense_formulate_alternative = define_chat_llm_call(
    [system_message(_alternative_dense_formulation_prompt_tmpl), user_message("{{search_query}}")],
    format="mustache",
    input_type=FormulateDenseQueriesInput,
    output_type=DenseQueries,
).map(lambda o: o.alternative_queries)

dense_formulate_prompts = [dense_formulate, dense_formulate_alternative]

# ----------------------- #
# ReFormulate Dense Query #
# ----------------------- #

_dense_reformulate_prompt_tmpl = """
Please formulate queries to run on a dense retrieval index to answer \
the following search query:
{search_query}.

The dense retrieval index contains full-text research papers taken from the arXiv and ACL Anthology.
Each passage in the dense index is a single sentence.
Note that the full-text of the papers is available (not only titles/abstracts), so when helpful you can query about \
specific or small details that may indicate relevance to the search query.

The following papers were found relevant.

Use the content of the relevant papers to formulate at most {max_output} different queries that will help find further \
relevant documents. Try to be creative and not repeat the previous queries.

Relevant Papers:
{papers}

Remember, your task is formulate queries to run on the dense retrieval index to answer the following search query:
{search_query}.
"""  # noqa: E501


class ReformulateDenseQueriesInput(FormulateDenseQueriesInput):
    papers: str


dense_reformulate = define_prompt_llm_call(
    _dense_reformulate_prompt_tmpl, input_type=ReformulateDenseQueriesInput, output_type=DenseQueries
).map(lambda o: o.alternative_queries)
