from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, TypedDict, get_args

from ai2i.chain import (
    ChainComputation,
    define_chat_llm_call,
    define_prompt_llm_call,
    system_message,
    user_message,
)
from ai2i.dcollection import (
    ExtractedYearlyTimeRange,
    RelevanceCriteria,
    RelevanceCriterion,
)
from pydantic import BaseModel, model_validator

from mabool.agents.common.common import (
    InputQuery,
    InputQueryJson,
    as_input_query_json,
)
from mabool.data_model.agent import (
    BroadOrSpecificType,
    ByNameOrTitleType,
    DomainsIdentified,
    ExtractedAuthors,
    ExtractedCentrality,
    ExtractedContent,
    ExtractedFields,
    ExtractedRecency,
    ExtractedVenues,
    FieldOfStudy,
    PossibleRefusal,
)
from mabool.data_model.specifications import Specifications

# ---------------------- #
# Suitable For Citations #
# ---------------------- #


class SuitableForByCitingInput(TypedDict):
    query: str
    extracted_name: str


class SuitableForByCiting(BaseModel):
    answer: bool


_suitable_for_by_citing_prompt_tmpl = """
Are the exact sentences where papers cite "the {extracted_name} paper" likely to contain highly relevant results for the following query?
{query}
"""  # noqa: E501


suitable_for_citing = define_prompt_llm_call(
    _suitable_for_by_citing_prompt_tmpl,
    input_type=SuitableForByCitingInput,
    output_type=SuitableForByCiting,
    custom_format_instructions=('Output a json with key "answer" with a boolean answer.'),
).map(lambda o: o.answer)


# ------------- #
# Check Refusal #
# ------------- #


_check_refusal_prompt_tmpl = """
You are a chatbot for finding academic papers, but there are a few things you don't support.
Given the below user query, identify if any of the following unsupported categories is relevant:

1. "not paper finding": when the query does not directly ask to find papers. or in other words the expected output is not a paper/resource or a list of papers/resources (and notice Note 1.3). e.g. "execute the algorithm from 'An Efficient Sorting Algorithm for Non-Volatile Memory' on this list ...". BUT be lenient in the following ways:
    1.1 if you are not sure, or the query is too complicated - allow it.
    1.2 "not paper finding" should take precedence over "web access" and "similar to"
    1.3 allow general open-ended questions, or queries seeking resources/data, recommendations, specific paper/work/resource search - if academic literature might help answer.
2. "similar to": when the searchable semantic criterion is missing and instead the query relies only on similarity to another papers' content. e.g. "find me papers that are like the BLAH paper"
    2.1 allow queries that ask about a paper citing or being cited by other papers, this does not fall under "similar to".
3. "web access": when the query explicitly(!) requires web access in order to find the answer. e.g. "find me the paper in the following url: 'https://bla.org/123456789'", or "find me the papers appearing in the latest Allen Institute's blog post"
4. "affiliation": we support filtering by author names, but do not support affiliation, organizations or groups. e.g. "find papers about SOMETHING written at META AI"
5. "author ID": we support filtering by author names, but do not support author IDs. e.g. "find me papers by John Doe (12345678)"

Use the category names as listed, or if none of them is relevant return null.
Output a refusal type in a Json like: {"type": string or null}
"""

_check_refusal = define_chat_llm_call(
    [system_message(_check_refusal_prompt_tmpl), user_message("{{query}}")],
    format="mustache",
    input_type=InputQueryJson,
    output_type=PossibleRefusal,
)


# --------------------- #
# Domain identification #
# --------------------- #


_domain_identification_prompt_tmpl = (
    """
Given a paper finding query, which of the following fields of study does it fall under?

Fields of study: """
    + ", ".join(get_args(FieldOfStudy))
    + """

Extract all possible fields of study, but divide it to a single most prominent main field and a (possibly empty) list of other relevant key fields.
Use Unknown only if no field-of-study is assignable.

{{query_json}}
"""
)

_domain_identification = define_prompt_llm_call(
    _domain_identification_prompt_tmpl,
    format="mustache",
    input_type=InputQueryJson,
    output_type=DomainsIdentified,
)


# ------------------ #
# Content Extraction #
# ------------------ #
_content_extraction_prompt_tmpl = """
# Task Definition

Given a query for finding papers about a specific topic, extract only the *content* of the query, ignoring all *metadata*. Metadata is defined to be either of the following:

* Author/Coauthor name(s)
* Year(s), or words that describe time, such as "recent", "latest"
* Words that describe the impact of the paper, such as "central", "seminal", "influential"
* Venues, such as ACL, EMNLP, AAAI
* Words that describe how the search should be carried out, such as "run an exhaustive search on..."

## Rules

* If the query contains phrases such as "papers using", "papers proposing", "survey on", keep them as part of the content. However, "papers about" and "papers on" can be ignored.
* If the query is in the form of a question, extract a coherent representation of the question that focuses on the content of the question.
* Queries that only contain metadata as described above should return `null`.
* If you're unsure what the content is, return the original query as-is. If there is no content, return `null`.

The return format should be a JSON object that looks like {"content": ...}

# Examples

{"query": "Graph-based Neural Multi-Document Summarization Yasunaga et al., 2017"}
{"content": "Graph-based Neural Multi-Document Summarization"}
Reason: "Yasunaga et al." is author metadata, and 2017 is time metadata

{"query": "classic or early papers on pretrained transformer models"}
{"content": "pretrained transformer models"}
Reason: "classic" is impact metadata, and "early" is time metadata

{"query": "good paper about CRISPR gene editing"}
{"content": "CRISPR gene editing"}
Reason: The word "good" doesn't modify the content and is inconsequential for the query

{"query": "papers about LLM chains"}
{"content": "LLM chains"}
Reason: Since the query is already for finding papers, the prefix "papers about" is redundant

{"query": "multi document summarization methods"}
{"content": "multi document summarization methods"}
Reason: Every word is essential to understand the query, and there's no metadata at all

{"query": "latest research on using annotation disagreements in classification models"}
{"content": "using annotation disagreements in classification models"}
Reason: "latest research" is time metadata

{"query": "papers from ICLR 2024"}
{"content": ""}
Reason: The query consists of metadata only"""  # noqa: E501

_content_extraction = define_chat_llm_call(
    [system_message(_content_extraction_prompt_tmpl), user_message("{{query_json}}")],
    format="mustache",
    input_type=InputQueryJson,
    output_type=ExtractedContent,
)


# ----------------- #
# Author Extraction #
# ----------------- #

_author_extraction_prompt_tmpl = """
# Task Definition

Given a query for finding papers, identify the authors whose papers are being requested.
Extract only the *author(s)* names from the query, ignoring all other information.

The return format should be a JSON object that looks like {"authors": [...]}.
If there are no authors' names requested in the query, return {"authors": []}.

# Examples

{"query": "Graph-based Neural Multi-Document Summarization Yasunaga et al., 2017"}
{"authors": ["Yasunaga"]}
Reason: The "et al." suffix does not provide information on the authors

{"query": "papers on planning by Dan Weld"}
{"authors": ["Dan Weld"]}

{"query": "papers about transformer models by Google"}
{"authors": []}
Reason: "Google" is an organization, not an author name. The query does not require papers from a specific author.

{"query": "papers by author with scopus ID 123456789"}
{"authors": []}
Reason: The query does not require papers from a specific author's name, but rather from a specific author ID.

{"query": "papers on LLM chains"}
{"authors": []}
Reason: The query does not require papers from a specific author.

{"query": "papers discussing Henry David Thoreau's ideas about nature"}
{"authors": []}
Reason: The author mentioned is not requested to be the author of the paper, but rather a person whose ideas should be discussed in the paper.
"""

_author_extraction = define_chat_llm_call(
    [system_message(_author_extraction_prompt_tmpl), user_message("{{query_json}}")],
    format="mustache",
    input_type=InputQueryJson,
    output_type=ExtractedAuthors,
)


# ----------------- #
# Venue Extraction #
# ----------------- #

_venue_extraction_prompt_tmpl = """
# Task Definition

Given a query for finding papers, what venue(s) does it require the papers to be from?
Extract only the *venue(s)* required in the query, ignoring all other information.

The return format should be a JSON object that looks like {"venues": [...]}. If there are no required venues in the query, return {"venues": []}.

# Examples

{"query": "Large Language Models can Strategically Deceive their Users when Put Under Pressure, ICLR 2024"}
{"venues": ["ICLR"]}

{"query": "papers presented at either ICLR or AAAI"}
{"venues": ["ICLR", "AAAI"]}

{"query": "papers that evaluate on the CoNLL-2003 benchmark"}
{"venues": []}
Reason: The query does not require papers from a specific venue. CoNLL-2003 in this case is NOT a required venue.

{"query": "ACL papers on transformers"}
{"venues": ["ACL", "EMNLP", "NAACL", "COLING", "EACL", "TACL", "CL", "LREC", "AACL", "CoNLL", "*SEM"]}
Reason: ACL may refer to a collection of venues, try to provide a comprehensive list of specific venues if possible.
"""  # noqa: E501

_venue_extraction = define_chat_llm_call(
    [system_message(_venue_extraction_prompt_tmpl), user_message("{{query_json}}")],
    format="mustache",
    input_type=InputQueryJson,
    output_type=ExtractedVenues,
)


# ------------------ #
# Recency Extraction #
# ------------------ #

_recency_extraction_prompt_tmpl = """
# Task Definition

Given a query for finding papers about a specific topic, decide whether the query explicitly asks for recent papers or early papers, ignoring all other information.

Do not assume recency based on absolute years.

If the query asks for most recent papers, return the JSON object {"prefer": "recent"}, if the query asks for early papers, return the JSON object {"prefer": "early"}, and otherwise return {"prefer": null}.

# Examples

{"query": "papers that have referenced 'Attention is All You Need' recently"}
{"prefer": "recent"}

{"query": "latest papers about claim verification"}
{"prefer": "recent"}

{"query": "earlier papers on seq2seq"}
{"prefer": "early"}

{"query": "survey on multi-agent collaboration in AI and HCI"}
{"prefer": null}
"""  # noqa: E501


class RecentOrEarlyType(BaseModel):
    prefer: Literal["recent", "early"] | None


def _map_extracted_recency(r: RecentOrEarlyType) -> ExtractedRecency:
    if r.prefer == "recent":
        return ExtractedRecency(recency="first")
    if r.prefer == "early":
        return ExtractedRecency(recency="last")
    return ExtractedRecency(recency=None)


_recency_extraction = define_chat_llm_call(
    [system_message(_recency_extraction_prompt_tmpl), user_message("{{query_json}}")],
    format="mustache",
    input_type=InputQueryJson,
    output_type=RecentOrEarlyType,
).map(_map_extracted_recency)


# --------------------- #
# Centrality Extraction #
# --------------------- #

_centrality_extraction_prompt_tmpl = """
# Task Definition

Given a query for finding papers about a specific topic, decide whether the query asks for central papers or less cited papers, ignoring all other information.

A query asks for a central paper if it uses words like "central", "seminal", "impactful", "highly influential", "highly cited", etc.

A query asks for a less cited paper if it uses words like "less cited", "lesser known", etc.

If the query asks for central papers, return the JSON object {"centrality": "first"}, if the query asks for less cited papers, return the JSON object {"centrality": "last"}, otherwise return {"centrality": null}.

# Examples

{"query": "most important references on counterfactual data augmentation (CDA)"}
{"centrality": "first"}

{"query": "top papers in AI for Earth (environmental AI)"}
{"centrality": "first"}

{"query": "least cited papers on transformers"}
{"centrality": "last"}

{"query": "papers on LSTMs that are the least cited"}
{"centrality": "last"}

{"query": "paper on weather"}
{"centrality": null}"""  # noqa: E501

_centrality_extraction = define_chat_llm_call(
    [system_message(_centrality_extraction_prompt_tmpl), user_message("{{query_json}}")],
    format="mustache",
    input_type=InputQueryJson,
    output_type=ExtractedCentrality,
)


# --------------------- #
# Time Range Extraction #
# --------------------- #

_time_range_prompt_tmpl = """
# Task Definition

The current year is 2025. Given a query for finding papers about a specific topic, extract the time range mentioned in the query, if it exists. Only extract explicit mentions of time ranges.

Return a JSON object in the format: {"start": ..., "end": ...}. If neither field exists, return {"start": null, "end": null}.

# Examples

{"query": "recent papers using Earth Mover's Distance (EMD) as an evaluation metric"}
{"start": null, "end": null}

{"query": "synthesizing answers to scientific questions from search or ranker result snippets or documents, multi-document answers synthesis, last 3 years"}
{"start": 2023, "end": 2025}
Reason: the last 3 years are 2025, 2024, and 2023

{"query": "research on persona-assigned Large Language Models published in 2024"}
{"start": 2024, "end": 2024}"""  # noqa: E501

_time_range = define_chat_llm_call(
    [system_message(_time_range_prompt_tmpl), user_message("{{query_json}}")],
    format="mustache",
    input_type=InputQueryJson,
    output_type=ExtractedYearlyTimeRange,
)


# ---------------------------- #
# Broad or Specific Extraction #
# ---------------------------- #

_broad_or_specific_query_type_prompt_tmpl = """
# Task Definition

You are given a user's query aimed at finding academic papers. Your goal is to determine the nature of the query based on the following criteria:

- ** unique-identifier **: The user knows the exact paper they are looking for and provides a unique identifier (the paper title or another unique name that uniquely identifies the paper).

- ** descriptions-or-keywords **: The user is searching for papers using a description, keywords, or topics. The user does not provide a unique identifier for the paper. The query may contain named entities, but they do not uniquely identify a specific paper the user is looking for.

If the query is searching by unique-identifier, return the JSON object {"type": "unique-identifier"}, otherwise return {"type": "descriptions-or-keywords"}.

# Examples

{"query": "llm hallucinations"}
{"type": "descriptions-or-keywords"}

{"query": "the snli paper"}
{"type": "unique-identifier"}

{"query": "Attention is All You Need"}
{"type": "unique-identifier"}

{"query": "GLUE paper about the evaluation of natural language understanding systems"}
{"type": "unique-identifier"}
Reason: The query is looking for a unique-identifier paper by name, and provides a description for extra context

{"query": "pretrained large language models"}
{"type": "descriptions-or-keywords"}

{"query": "paper showing that transformers are better than LSTMs"}
{"type": "descriptions-or-keywords"}

{"query": "the first paper that evaluated the performance of transformers on the GLUE benchmark"}
{"type": "descriptions-or-keywords"}
Reason: The query is looking for a specific paper, but the user does not provide a unique identifier. None of the names provided ("GLUE", "transformer") uniquely identify the paper the user is looking for.
"""

UniqueOrDesc = Literal["unique-identifier", "descriptions-or-keywords"]


class UniqueOrDescType(BaseModel):
    type: UniqueOrDesc


def _to_broad_or_specific_type(udt: UniqueOrDescType) -> BroadOrSpecificType:
    return (
        BroadOrSpecificType(type="specific") if udt.type == "unique-identifier" else BroadOrSpecificType(type="broad")
    )


_broad_or_specific_query_type = define_chat_llm_call(
    [system_message(_broad_or_specific_query_type_prompt_tmpl), user_message("{{query_json}}")],
    format="mustache",
    input_type=InputQueryJson,
    output_type=UniqueOrDescType,
).map(_to_broad_or_specific_type)

# --------------------------- #
# By Title Or Name Extraction #
# --------------------------- #

_by_title_or_name_query_type_prompt_tmpl = """
# Task Definition

Given a query for finding a specific paper, decide whether the query is looking for a paper by its title, or by some key features.

If the query is looking for a paper by name, return the JSON object {"type": "title"}, otherwise return {"type": "name"}. If unsure, return {"type": "name"}.

# Examples

{"query": "Attention is All You Need"}
{"type": "title"}

{"query": "BioBERT: a pre-trained biomedical language representation model for biomedical text mining"}
{"type": "title"}

{"query": "the snli paper"}
{"type": "name"}

{"query": "LEGOBench dataset"}
{"type": "name"}"""  # noqa: E501

_by_title_or_name_query_type = define_chat_llm_call(
    [system_message(_by_title_or_name_query_type_prompt_tmpl), user_message("{{query_json}}")],
    format="mustache",
    input_type=InputQueryJson,
    output_type=ByNameOrTitleType,
)


_identify_relevance_criteria_prompt_tmpl = """
Identify a set of criteria for relevance for the following query.
These will later be used to judge the relevance of specific papers, and then filter and rank by the judgements.
Make sure not to lose necessary relations between criteria.
- A simple example of such failure: for the query "LLMs for NLP" the criteria "LLMs" and "NLP" are identified, but the relation between them is lost. So papers that mention LLMs and NLP, but not *LLMs for NLP*, are considered relevant.
- Do this either by adding a heavily-weighted criterion that captures a required relation between other criteria, or by adding the required relation to each description, but keeping its main focus.

The criteria should only refer to the content of the paper, ignoring all metadata criteria. Metadata is defined to be either of the following:
* Author/Coauthor name(s)
* Year(s), or words that describe time, such as "recent", "latest"
* Words that describe the impact of the paper, such as "central", "seminal", "influential"
* Venues, such as ACL, EMNLP, AAAI.

Output a json with key "required_relevance_critieria", with its value a list of criteria.
Each criterion must have a "name" key, a "description" key, and a "weight" key (0-1). The sum of these weights should be equal to 1.
"required_relevance_critieria" must appear in the output json.
The json can contain another key "nice_to_have_relevance_criteria", with the same type of value, only here the weights don't have to sum to 1.

If the query is too ambiguous to devise a good set of criteria, please ask clarification questions.
In this case the json will contain a single key "clarification_questions", with it's value a list of string questions.
Only ask necessary questions, and try to keep them as simple as possible.
"""


class IdentifyRelevanceCriteriaOutput(BaseModel):
    required_relevance_critieria: Optional[list[RelevanceCriterion]] = None
    nice_to_have_relevance_criteria: Optional[list[RelevanceCriterion]] = None
    clarification_questions: Optional[list[str]] = None

    @model_validator(mode="after")
    def check_at_least_one_not_none(self) -> IdentifyRelevanceCriteriaOutput:
        if self.required_relevance_critieria is None and self.clarification_questions is None:
            raise ValueError(
                "At least one of 'required_relevance_critieria' or 'clarification_questions' must be provided."
            )
        return self

    @model_validator(mode="after")
    def check_all_criterion_names_are_distinct(self) -> IdentifyRelevanceCriteriaOutput:
        criterion_names = set()
        if self.required_relevance_critieria is not None:
            criterion_names |= {criterion.name for criterion in self.required_relevance_critieria}
        if self.nice_to_have_relevance_criteria is not None:
            criterion_names |= {criterion.name for criterion in self.nice_to_have_relevance_criteria}

        if len(criterion_names) != len(self.required_relevance_critieria or []) + len(
            self.nice_to_have_relevance_criteria or []
        ):
            raise ValueError("Criterion names must be distinct.")
        return self

    @model_validator(mode="after")
    def check_weights_sum_to_one(self) -> IdentifyRelevanceCriteriaOutput:
        if self.required_relevance_critieria is not None:
            if sum(criterion.weight for criterion in self.required_relevance_critieria) != 1:
                raise ValueError("The sum of weights for required relevance criteria must be 1.")
        return self


def _map_relevance_criteria_output(
    t: tuple[InputQueryJson, IdentifyRelevanceCriteriaOutput],
) -> RelevanceCriteria:
    query, output = t
    return RelevanceCriteria(
        query=query["query"],
        required_relevance_critieria=output.required_relevance_critieria,
        nice_to_have_relevance_criteria=output.nice_to_have_relevance_criteria,
        clarification_questions=output.clarification_questions,
    )


_identify_relevance_criteria = (
    define_chat_llm_call(
        [system_message(_identify_relevance_criteria_prompt_tmpl), user_message("{{query}}")],
        format="mustache",
        input_type=InputQueryJson,
        output_type=IdentifyRelevanceCriteriaOutput,
    )
    .passthrough_input()
    .map(_map_relevance_criteria_output)
)


# ------------------------ #
# Specification extraction #
# ------------------------ #

_my_dir = Path(__file__).parent
extract_specifications_path = _my_dir / "extract_specifications.md"
with open(extract_specifications_path, "r") as fp:
    _extract_specification_prompt = fp.read()
specification_extraction = define_chat_llm_call(
    [
        system_message(_extract_specification_prompt),
        user_message("{{query}}"),
    ],
    format="mustache",
    input_type=InputQueryJson,
    output_type=Specifications,
).contra_map(as_input_query_json)


# --------------------- #
# Full Decompose prompt #
# --------------------- #


def _combine_extracted_fields(
    content: ExtractedContent,
    authors: ExtractedAuthors,
    venues: ExtractedVenues,
    recency: ExtractedRecency,
    centrality: ExtractedCentrality,
    time_range: ExtractedYearlyTimeRange,
    broad_or_specific: BroadOrSpecificType,
    by_name_or_title: ByNameOrTitleType,
    relevance_criteria: RelevanceCriteria,
    domains: DomainsIdentified,
    possible_refusal: PossibleRefusal,
) -> ExtractedFields:
    return {
        "content": content,
        "authors": authors,
        "venues": venues,
        "recency": recency,
        "centrality": centrality,
        "time_range": time_range,
        "broad_or_specific": broad_or_specific,
        "by_name_or_title": by_name_or_title,
        "relevance_criteria": relevance_criteria,
        "domains": domains,
        "possible_refusal": possible_refusal,
    }


decompose_query = ChainComputation.map_n(
    _combine_extracted_fields,
    _content_extraction,
    _author_extraction,
    _venue_extraction,
    _recency_extraction,
    _centrality_extraction,
    _time_range,
    _broad_or_specific_query_type,
    _by_title_or_name_query_type,
    _identify_relevance_criteria,
    _domain_identification,
    _check_refusal,
).contra_map(as_input_query_json)


# ------------------------ #
# Resource Name Extraction #
# ------------------------ #


class NameExtractionInput(InputQuery):
    domain_description: str
    determiner_example: str
    affiliation_example: str


class ExtractedName(BaseModel):
    name: str | None
    corrected: str | None


def _get_extracted_name(e: ExtractedName) -> str | None:
    return e.corrected if e.corrected else e.name


_name_extraction_prompt_tmpl = """
Extract the name of the most central resource the following query is looking for:
{query}

This could be a method name, technique name, dataset name, model name, concept, or any other type of name mentioned in the query.

Do not include any additional information such as URLs, titles, authors, bibtex shorthands, years, or other metadata.
Only include the name, not determiners like "the" and not qualifiers such as {determiner_example}.
In cases where there is a name and affiliation mentioned, extract only the name (for example {affiliation_example}).

Keep the original capitalization and spelling of the name in the "name" field.
If the capitalization or spelling is incorrect, provide the commonly used version in the "corrected" field.
"""  # noqa: E501

name_extraction = define_prompt_llm_call(
    _name_extraction_prompt_tmpl,
    input_type=NameExtractionInput,
    output_type=ExtractedName,
    custom_format_instructions=(
        'Output a json with key "name" with the extracted name and an optional "corrected" key.'
    ),
).map(_get_extracted_name)
