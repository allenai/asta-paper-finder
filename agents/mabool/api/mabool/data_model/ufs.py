from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai2i.config.config_models import ConfigValuePlaceholder


@dataclass(frozen=True)
class NoPerfectMatches:
    intro: ConfigValuePlaceholder[Literal["While I didn't find a perfect match in this search, I did find "]] = (
        ConfigValuePlaceholder(["explain", "merged_summary", "no_perfect_matches", "intro"])
    )
    more_highly_relevant_singular: ConfigValuePlaceholder[Literal["one additional paper that is relevant"]] = (
        ConfigValuePlaceholder(["explain", "merged_summary", "no_perfect_matches", "more_highly_relevant_singular"])
    )
    more_highly_relevant_plural: ConfigValuePlaceholder[Literal["{count} additional relevant ones"]] = (
        ConfigValuePlaceholder(["explain", "merged_summary", "no_perfect_matches", "more_highly_relevant_plural"])
    )
    more_potentially_relevant_singular: ConfigValuePlaceholder[
        Literal["one additional paper that is potentially relevant"]
    ] = ConfigValuePlaceholder(
        ["explain", "merged_summary", "no_perfect_matches", "more_potentially_relevant_singular"]
    )
    more_potentially_relevant_plural: ConfigValuePlaceholder[Literal["{count} potentially relevant ones"]] = (
        ConfigValuePlaceholder(["explain", "merged_summary", "no_perfect_matches", "more_potentially_relevant_plural"])
    )


@dataclass(frozen=True)
class PotentiallyRelevant:
    singular: ConfigValuePlaceholder[
        Literal[" I've also found one additional paper that potentially matches your query."]
    ] = ConfigValuePlaceholder(["explain", "merged_summary", "potentially_relevant", "singular"])
    plural: ConfigValuePlaceholder[
        Literal[" I've also found {count} additional papers that potentially match your query."]
    ] = ConfigValuePlaceholder(["explain", "merged_summary", "potentially_relevant", "plural"])


@dataclass(frozen=True)
class PreviousSearch:
    perfect_plural: ConfigValuePlaceholder[Literal["Your previous search yielded {count} perfect matches."]] = (
        ConfigValuePlaceholder(["explain", "merged_summary", "previous_search", "perfect_plural"])
    )
    perfect_singular: ConfigValuePlaceholder[Literal["Your previous search yielded one perfect match."]] = (
        ConfigValuePlaceholder(["explain", "merged_summary", "previous_search", "perfect_singular"])
    )
    no_perfect: ConfigValuePlaceholder[Literal["Your previous search yielded no perfect matches."]] = (
        ConfigValuePlaceholder(["explain", "merged_summary", "previous_search", "no_perfect"])
    )
    no_additional: ConfigValuePlaceholder[Literal[" In this search I found no additional matches."]] = (
        ConfigValuePlaceholder(["explain", "merged_summary", "previous_search", "no_additional"])
    )


@dataclass(frozen=True)
class NoPerfect:
    intro: ConfigValuePlaceholder[Literal["While I didn't find a perfect match, I did find "]] = ConfigValuePlaceholder(
        ["explain", "summary", "no_perfect", "intro"]
    )
    highly_relevant_only_plural: ConfigValuePlaceholder[Literal["{count} relevant ones"]] = ConfigValuePlaceholder(
        ["explain", "summary", "no_perfect", "highly_relevant_only_plural"]
    )
    highly_relevant_only_singular: ConfigValuePlaceholder[Literal["one that is relevant"]] = ConfigValuePlaceholder(
        ["explain", "summary", "no_perfect", "highly_relevant_only_singular"]
    )
    potentially_relevant_plural: ConfigValuePlaceholder[Literal["{count} potentially relevant ones"]] = (
        ConfigValuePlaceholder(["explain", "summary", "no_perfect", "potentially_relevant_plural"])
    )
    potentially_relevant_singular: ConfigValuePlaceholder[Literal["one that is potentially relevant"]] = (
        ConfigValuePlaceholder(["explain", "summary", "no_perfect", "potentially_relevant_singular"])
    )
    and_others: ConfigValuePlaceholder[Literal[" and {count} {others_label}"]] = ConfigValuePlaceholder(
        ["explain", "summary", "no_perfect", "and_others"]
    )


@dataclass(frozen=True)
class OtherRelevant:
    default: ConfigValuePlaceholder[Literal[" and {others}"]] = ConfigValuePlaceholder(
        ["explain", "summary", "other_relevant", "default"]
    )


@dataclass(frozen=True)
class HighlyRelevant:
    plural: ConfigValuePlaceholder[Literal["{connector}{count} relevant ones"]] = ConfigValuePlaceholder(
        ["explain", "summary", "highly_relevant", "plural"]
    )
    singular: ConfigValuePlaceholder[Literal["{connector}{count} that is relevant"]] = ConfigValuePlaceholder(
        ["explain", "summary", "highly_relevant", "singular"]
    )


@dataclass(frozen=True)
class PerfectMatches:
    plural: ConfigValuePlaceholder[Literal["I found {count} papers that look like perfect matches"]] = (
        ConfigValuePlaceholder(["explain", "summary", "perfect_matches", "plural"])
    )
    singular: ConfigValuePlaceholder[Literal["I found one paper that looks like a perfect match"]] = (
        ConfigValuePlaceholder(["explain", "summary", "perfect_matches", "singular"])
    )


@dataclass(frozen=True)
class Explanation:
    most_recent: ConfigValuePlaceholder[Literal["most-recent"]] = ConfigValuePlaceholder(
        ["sorting", "explanation", "most_recent"]
    )
    earliest: ConfigValuePlaceholder[Literal["earliest"]] = ConfigValuePlaceholder(
        ["sorting", "explanation", "earliest"]
    )
    highly_cited: ConfigValuePlaceholder[Literal["highly-cited"]] = ConfigValuePlaceholder(
        ["sorting", "explanation", "highly_cited"]
    )
    least_cited: ConfigValuePlaceholder[Literal["least-cited"]] = ConfigValuePlaceholder(
        ["sorting", "explanation", "least_cited"]
    )
    sorting_explanation: ConfigValuePlaceholder[
        Literal["Between relevant papers, {sorting_explanation} papers were prioritized due to explicit request."]
    ] = ConfigValuePlaceholder(["sorting", "explanation", "sorting_explanation"])


@dataclass(frozen=True)
class MergedSummary:
    previous_search: PreviousSearch = PreviousSearch()
    potentially_relevant: PotentiallyRelevant = PotentiallyRelevant()
    perfect_matches: PerfectMatches = PerfectMatches()
    no_perfect_matches: NoPerfectMatches = NoPerfectMatches()
    perfect_intro: ConfigValuePlaceholder[Literal[" I've also found "]] = ConfigValuePlaceholder(
        ["explain", "merged_summary", "perfect_intro"]
    )


@dataclass(frozen=True)
class Summary:
    perfect_matches: PerfectMatches = PerfectMatches()
    highly_relevant: HighlyRelevant = HighlyRelevant()
    other_relevant: OtherRelevant = OtherRelevant()
    no_perfect: NoPerfect = NoPerfect()
    no_results: ConfigValuePlaceholder[Literal["I didn't find papers matching your query."]] = ConfigValuePlaceholder(
        ["explain", "summary", "no_results"]
    )
    found_papers: ConfigValuePlaceholder[Literal["I found {paper_count} matching your query."]] = (
        ConfigValuePlaceholder(["explain", "summary", "found_papers"])
    )
    potential_matches: ConfigValuePlaceholder[Literal["I found {paper_count} that potentially match your query."]] = (
        ConfigValuePlaceholder(["explain", "summary", "potential_matches"])
    )


@dataclass(frozen=True)
class Domains:
    main_field: ConfigValuePlaceholder[Literal["\t- Main FoS: {field}"]] = ConfigValuePlaceholder(
        ["explain", "domains", "main_field"]
    )
    secondary_fields: ConfigValuePlaceholder[Literal["\t- Key Secondary FoS: {fields}"]] = ConfigValuePlaceholder(
        ["explain", "domains", "secondary_fields"]
    )


@dataclass(frozen=True)
class ContentSuffix:
    with_content: ConfigValuePlaceholder[Literal[", also discussing: {content}"]] = ConfigValuePlaceholder(
        ["explain", "content_suffix", "with_content"]
    )


@dataclass(frozen=True)
class Content:
    specific_paper_by_name: ConfigValuePlaceholder[Literal["- A paper with the alias: '{name}'{content_suffix}."]] = (
        ConfigValuePlaceholder(["explain", "content", "specific_paper_by_name"])
    )
    specific_paper_by_title: ConfigValuePlaceholder[Literal["- A paper titled: '{title}'."]] = ConfigValuePlaceholder(
        ["explain", "content", "specific_paper_by_title"]
    )
    papers_discussing: ConfigValuePlaceholder[Literal["- Papers discussing: *{content}*."]] = ConfigValuePlaceholder(
        ["explain", "content", "papers_discussing"]
    )
    relevance_criteria_header: ConfigValuePlaceholder[Literal["- Judged by the following relevance criteria:"]] = (
        ConfigValuePlaceholder(["explain", "content", "relevance_criteria_header"])
    )
    criterion_format: ConfigValuePlaceholder[Literal["\t- *{name}*: {description}"]] = ConfigValuePlaceholder(
        ["explain", "content", "criterion_format"]
    )
    fields_of_study_header: ConfigValuePlaceholder[Literal["- Fields of Study:"]] = ConfigValuePlaceholder(
        ["explain", "content", "fields_of_study_header"]
    )
    metadata_header: ConfigValuePlaceholder[Literal["- Metadata:"]] = ConfigValuePlaceholder(
        ["explain", "content", "metadata_header"]
    )


@dataclass(frozen=True)
class Metadata:
    time_same_year: ConfigValuePlaceholder[Literal[" during {year}"]] = ConfigValuePlaceholder(
        ["explain", "metadata", "time_same_year"]
    )
    time_range_start_and_end: ConfigValuePlaceholder[Literal[" from {start} until {end}"]] = ConfigValuePlaceholder(
        ["explain", "metadata", "time_range_start_and_end"]
    )
    time_range_start_only: ConfigValuePlaceholder[Literal[" from {start}"]] = ConfigValuePlaceholder(
        ["explain", "metadata", "time_range_start_only"]
    )
    time_range_end_only: ConfigValuePlaceholder[Literal[" until {end}"]] = ConfigValuePlaceholder(
        ["explain", "metadata", "time_range_end_only"]
    )
    time_range_prefix: ConfigValuePlaceholder[Literal["\t- Time-range:{time}"]] = ConfigValuePlaceholder(
        ["explain", "metadata", "time_range_prefix"]
    )
    venue: ConfigValuePlaceholder[Literal["\t- Venue{s}: {items}"]] = ConfigValuePlaceholder(
        ["explain", "metadata", "venue"]
    )
    author: ConfigValuePlaceholder[Literal["\t- Author{s}: {items}"]] = ConfigValuePlaceholder(
        ["explain", "metadata", "author"]
    )


@dataclass(frozen=True)
class Refusal:
    similar_to: ConfigValuePlaceholder[
        Literal[
            "I'm sorry I do not support similarity queries at this point. I will soon. For now, you can ask specifically about the content you want me to help you find."
        ]
    ] = ConfigValuePlaceholder(["response_texts", "refusal", "similar_to"])
    web_access: ConfigValuePlaceholder[
        Literal[
            "I'm sorry I can not access the web. I can only search within the 200 million papers available in Semantic Scholar."
        ]
    ] = ConfigValuePlaceholder(["response_texts", "refusal", "web_access"])
    not_paper_finding: ConfigValuePlaceholder[
        Literal["I'm not sure what paper search will help answer your query."]
    ] = ConfigValuePlaceholder(["response_texts", "refusal", "not_paper_finding"])
    affiliation: ConfigValuePlaceholder[
        Literal["It seems your query contains affiliation criteria which I do not support yet."]
    ] = ConfigValuePlaceholder(["response_texts", "refusal", "affiliation"])
    author_id: ConfigValuePlaceholder[
        Literal["It seems your query contains author IDs criteria which I do not support yet."]
    ] = ConfigValuePlaceholder(["response_texts", "refusal", "author_id"])
    came_up: ConfigValuePlaceholder[Literal["This is the search I came up with:"]] = ConfigValuePlaceholder(
        ["response_texts", "refusal", "came_up"]
    )
    should_run: ConfigValuePlaceholder[Literal["Should I run it?"]] = ConfigValuePlaceholder(
        ["response_texts", "refusal", "should_run"]
    )
    yes_or_no: ConfigValuePlaceholder[
        Literal["Please answer with yes or no, as any other option is currently out of my scope."]
    ] = ConfigValuePlaceholder(["response_texts", "refusal", "yes_or_no"])


@dataclass(frozen=True)
class PaperFinderAgent:
    anything_else: ConfigValuePlaceholder[Literal["Would you like to try anything else?"]] = ConfigValuePlaceholder(
        ["response_texts", "paper_finder_agent", "anything_else"]
    )
    soft_rejection: ConfigValuePlaceholder[Literal["soft rejection due to '{refusal_type}' type of query"]] = (
        ConfigValuePlaceholder(["response_texts", "paper_finder_agent", "soft_rejection"])
    )
    choice_explanation_prefix: ConfigValuePlaceholder[
        Literal[
            "It seems the query would not benefit from a paper search, or is not fully supported. The user chose to"
        ]
    ] = ConfigValuePlaceholder(["response_texts", "paper_finder_agent", "choice_explanation_prefix"])
    run_anyway: ConfigValuePlaceholder[Literal["{choice_explanation} run the paper search anyway."]] = (
        ConfigValuePlaceholder(["response_texts", "paper_finder_agent", "run_anyway"])
    )
    cancel: ConfigValuePlaceholder[Literal["{choice_explanation} cancel the search."]] = ConfigValuePlaceholder(
        ["response_texts", "paper_finder_agent", "cancel"]
    )


@dataclass(frozen=True)
class MetadataAgent:
    could_not_find_in_s2: ConfigValuePlaceholder[
        Literal["I couldn't find any results for this query in the Semantic Scholar index."]
    ] = ConfigValuePlaceholder(["response_texts", "metadata_agent", "could_not_find_in_s2"])
    try_alternative: ConfigValuePlaceholder[Literal[" Perhaps try an alternative venue name?"]] = (
        ConfigValuePlaceholder(["response_texts", "metadata_agent", "try_alternative"])
    )
    notice_limit: ConfigValuePlaceholder[
        Literal[
            "Please notice, these are the first {limit} papers that matched this criteria. I'm sorry but I could not fetch more."
        ]
    ] = ConfigValuePlaceholder(["response_texts", "metadata_agent", "notice_limit"])


@dataclass(frozen=True)
class Snowball:
    following_citations: ConfigValuePlaceholder[
        Literal["Following citations that were mentioned in relevant passages."]
    ] = ConfigValuePlaceholder(["step_progress", "snowball", "following_citations"])
    looking_for_papers_that_cite: ConfigValuePlaceholder[Literal["Looking at papers that cite promising papers."]] = (
        ConfigValuePlaceholder(["step_progress", "snowball", "looking_for_papers_that_cite"])
    )
    looking_for_papers_cited: ConfigValuePlaceholder[
        Literal["Looking at papers that were cited by relevant papers."]
    ] = ConfigValuePlaceholder(["step_progress", "snowball", "looking_for_papers_cited"])


@dataclass(frozen=True)
class PaperFinder:
    sorting_resultset: ConfigValuePlaceholder[Literal["Sorting result set."]] = ConfigValuePlaceholder(
        ["step_progress", "paper_finder", "sorting_resultset"]
    )


@dataclass(frozen=True)
class SpecificPaperByTitle:
    attempting_to_fetch: ConfigValuePlaceholder[Literal['Attempting to fetch _"{extracted_title}"_.']] = (
        ConfigValuePlaceholder(["step_progress", "specific_paper_by_title", "attempting_to_fetch"])
    )


@dataclass(frozen=True)
class SpecificPaperByName:
    looking_for_citations: ConfigValuePlaceholder[Literal['Looking for citations near _"{extracted_name}"_.']] = (
        ConfigValuePlaceholder(["step_progress", "specific_paper_by_name", "looking_for_citations"])
    )
    attempting_to_locate: ConfigValuePlaceholder[Literal['Attempting to locate: _"{extracted_content}"_.']] = (
        ConfigValuePlaceholder(["step_progress", "specific_paper_by_name", "attempting_to_locate"])
    )
    searching_semantic_scholar: ConfigValuePlaceholder[Literal["Searching semantic scholar."]] = ConfigValuePlaceholder(
        ["step_progress", "specific_paper_by_name", "searching_semantic_scholar"]
    )


@dataclass(frozen=True)
class MetadataOnly:
    s2_metadata_search: ConfigValuePlaceholder[Literal["Performing Semantic Scholar metadata search."]] = (
        ConfigValuePlaceholder(["step_progress", "metadata_only", "s2_metadata_search"])
    )
    metadata_search: ConfigValuePlaceholder[Literal["Metadata search."]] = ConfigValuePlaceholder(
        ["step_progress", "metadata_only", "metadata_search"]
    )


@dataclass(frozen=True)
class Dense:
    expanding_search_criteria: ConfigValuePlaceholder[Literal['Expanding search criteria for _"{search_query}"_.']] = (
        ConfigValuePlaceholder(["step_progress", "dense", "expanding_search_criteria"])
    )
    searching_using_expanded_criteria: ConfigValuePlaceholder[Literal["Searching using expanded criteria."]] = (
        ConfigValuePlaceholder(["step_progress", "dense", "searching_using_expanded_criteria"])
    )
    running_similatiry_based_search: ConfigValuePlaceholder[Literal["Running similarity-based search"]] = (
        ConfigValuePlaceholder(["step_progress", "dense", "running_similatiry_based_search"])
    )


@dataclass(frozen=True)
class BroadSearch:
    apply_relevance_judgement: ConfigValuePlaceholder[Literal["Assessing relevance of snippets and documents."]] = (
        ConfigValuePlaceholder(["step_progress", "broad_search", "apply_relevance_judgement"])
    )
    search_for_papers: ConfigValuePlaceholder[Literal["Searching for papers (deep-search mode)."]] = (
        ConfigValuePlaceholder(["step_progress", "broad_search", "search_for_papers"])
    )


@dataclass(frozen=True)
class FastBroadSearch:
    running_search: ConfigValuePlaceholder[
        Literal['Running keyword and semantic searches for _"{content_query}"_.']
    ] = ConfigValuePlaceholder(["step_progress", "fast_broad_search", "running_search"])
    rerank_docs: ConfigValuePlaceholder[Literal["Reranking candidate documents."]] = ConfigValuePlaceholder(
        ["step_progress", "fast_broad_search", "rerank_docs"]
    )
    judging_relevance: ConfigValuePlaceholder[Literal["Judging relevance of top documents."]] = ConfigValuePlaceholder(
        ["step_progress", "fast_broad_search", "judging_relevance"]
    )
    searching_for_papers: ConfigValuePlaceholder[Literal["Searching for papers."]] = ConfigValuePlaceholder(
        ["step_progress", "fast_broad_search", "searching_for_papers"]
    )


@dataclass(frozen=True)
class BroadSearchByKeyword:
    searching_s2: ConfigValuePlaceholder[Literal['Searching Semantic Scholar for _"{query}"_.']] = (
        ConfigValuePlaceholder(["step_progress", "broad_search_by_keyword", "searching_s2"])
    )
    searching_by_keyword: ConfigValuePlaceholder[Literal["Searching by keyword."]] = ConfigValuePlaceholder(
        ["step_progress", "broad_search_by_keyword", "searching_by_keyword"]
    )


@dataclass(frozen=True)
class BroadBySpecificPaperCitation:
    could_not_find_paper: ConfigValuePlaceholder[Literal['Could not find a paper for _"{extracted_name}"_.']] = (
        ConfigValuePlaceholder(["step_progress", "broad_by_specific_paper_citation", "could_not_find_paper"])
    )
    attempting_to_associate: ConfigValuePlaceholder[
        Literal['Attempting to associate _"{extracted_name}"_ with a paper.']
    ] = ConfigValuePlaceholder(["step_progress", "broad_by_specific_paper_citation", "attempting_to_associate"])
    attempting_fetch_citing: ConfigValuePlaceholder[
        Literal["Found papers:\n{paper_titles}.\n\nAttempting to fetch citing papers."]
    ] = ConfigValuePlaceholder(["step_progress", "broad_by_specific_paper_citation", "attempting_fetch_citing"])


@dataclass(frozen=True)
class LlmSuggest:
    received_n_suggestions: ConfigValuePlaceholder[Literal["Received {n} suggestions."]] = ConfigValuePlaceholder(
        ["step_progress", "llm_suggest", "received_n_suggestions"]
    )
    asking_llm: ConfigValuePlaceholder[Literal["Asking an LLM for suggested papers."]] = ConfigValuePlaceholder(
        ["step_progress", "llm_suggest", "asking_llm"]
    )
    verifying: ConfigValuePlaceholder[Literal["Verifying the existence of {papers} on Semantic Scholar."]] = (
        ConfigValuePlaceholder(["step_progress", "llm_suggest", "verifying"])
    )


@dataclass(frozen=True)
class SearchByAuthors:
    found_authors: ConfigValuePlaceholder[Literal["Found authors:\n{author_names}."]] = ConfigValuePlaceholder(
        ["step_progress", "search_by_authors", "found_authors"]
    )
    could_not_find_authors: ConfigValuePlaceholder[Literal["Could not find authors."]] = ConfigValuePlaceholder(
        ["step_progress", "search_by_authors", "could_not_find_authors"]
    )
    fetching: ConfigValuePlaceholder[Literal["Fetching papers from {author_names}."]] = ConfigValuePlaceholder(
        ["step_progress", "search_by_authors", "fetching"]
    )
    assessing: ConfigValuePlaceholder[Literal["Assessing papers for relevance."]] = ConfigValuePlaceholder(
        ["step_progress", "search_by_authors", "assessing"]
    )
    searching_for_papers_by_authors: ConfigValuePlaceholder[Literal["Searching for papers by authors."]] = (
        ConfigValuePlaceholder(["step_progress", "search_by_authors", "searching_for_papers_by_authors"])
    )


@dataclass(frozen=True)
class GatherUserNeeds:
    analyzing_request: ConfigValuePlaceholder[Literal["Analyzing request."]] = ConfigValuePlaceholder(
        ["step_progress", "gather_user_needs", "analyzing_request"]
    )


@dataclass(frozen=True)
class RoundExecutor:
    new_topic: ConfigValuePlaceholder[Literal["Searching for new topic."]] = ConfigValuePlaceholder(
        ["step_progress", "round_executor", "new_topic"]
    )
    refinement: ConfigValuePlaceholder[Literal["Refining previous query."]] = ConfigValuePlaceholder(
        ["step_progress", "round_executor", "refinement"]
    )
    correction: ConfigValuePlaceholder[Literal["Correcting previous query."]] = ConfigValuePlaceholder(
        ["step_progress", "round_executor", "correction"]
    )
    work_harder: ConfigValuePlaceholder[Literal["Working harder on previous query."]] = ConfigValuePlaceholder(
        ["step_progress", "round_executor", "work_harder"]
    )
    processing_request: ConfigValuePlaceholder[Literal['Processing request: _"{query}"_.']] = ConfigValuePlaceholder(
        ["step_progress", "round_executor", "processing_request"]
    )
    initiating_work_harder: ConfigValuePlaceholder[Literal["Initiating work-harder request."]] = ConfigValuePlaceholder(
        ["step_progress", "round_executor", "initiating_work_harder"]
    )
    finalizing: ConfigValuePlaceholder[Literal["Finalizing response."]] = ConfigValuePlaceholder(
        ["step_progress", "round_executor", "finalizing"]
    )


@dataclass(frozen=True)
class ConversationAgent:
    start_new_topic: ConfigValuePlaceholder[Literal["Starting a new query."]] = ConfigValuePlaceholder(
        ["step_progress", "conversation_agent", "start_new_topic"]
    )
    refine: ConfigValuePlaceholder[Literal['Refining an earlier query "{query}".']] = ConfigValuePlaceholder(
        ["step_progress", "conversation_agent", "refine"]
    )
    correct: ConfigValuePlaceholder[Literal['Attempting to correct an earlier query "{query}".']] = (
        ConfigValuePlaceholder(["step_progress", "conversation_agent", "correct"])
    )
    work_harder: ConfigValuePlaceholder[Literal["Working harder on query."]] = ConfigValuePlaceholder(
        ["step_progress", "conversation_agent", "work_harder"]
    )
    work_harder_previous: ConfigValuePlaceholder[Literal["Working harder on previous query."]] = ConfigValuePlaceholder(
        ["step_progress", "conversation_agent", "work_harder_previous"]
    )
    getting_results: ConfigValuePlaceholder[Literal["Focusing on {range_desc}."]] = ConfigValuePlaceholder(
        ["step_progress", "conversation_agent", "getting_results"]
    )
    getting_authors: ConfigValuePlaceholder[Literal["Looking at authors for {range_desc}."]] = ConfigValuePlaceholder(
        ["step_progress", "conversation_agent", "getting_authors"]
    )
    getting_titles: ConfigValuePlaceholder[Literal["Looking at titles for {range_desc}."]] = ConfigValuePlaceholder(
        ["step_progress", "conversation_agent", "getting_titles"]
    )
    getting_abstracts: ConfigValuePlaceholder[Literal["Looking at abstracts for {range_desc}."]] = (
        ConfigValuePlaceholder(["step_progress", "conversation_agent", "getting_abstracts"])
    )
    getting_years: ConfigValuePlaceholder[Literal["Looking at years for {range_desc}."]] = ConfigValuePlaceholder(
        ["step_progress", "conversation_agent", "getting_years"]
    )
    getting_venues: ConfigValuePlaceholder[Literal["Looking at venues for {range_desc}."]] = ConfigValuePlaceholder(
        ["step_progress", "conversation_agent", "getting_venues"]
    )
    getting_citation_counts: ConfigValuePlaceholder[Literal["Looking at citation counts for {range_desc}."]] = (
        ConfigValuePlaceholder(["step_progress", "conversation_agent", "getting_citation_counts"])
    )
    getting_relevance: ConfigValuePlaceholder[Literal["Looking at relevance for {range_desc}."]] = (
        ConfigValuePlaceholder(["step_progress", "conversation_agent", "getting_relevance"])
    )
    getting_relevance_summaries: ConfigValuePlaceholder[Literal["Looking at relevance summaries for {range_desc}."]] = (
        ConfigValuePlaceholder(["step_progress", "conversation_agent", "getting_relevance_summaries"])
    )
    finalizing_response: ConfigValuePlaceholder[Literal["Finalizing response."]] = ConfigValuePlaceholder(
        ["step_progress", "conversation_agent", "finalizing_response"]
    )


@dataclass(frozen=True)
class Verbalize:
    authored_by: ConfigValuePlaceholder[Literal["Authored by {authors}."]] = ConfigValuePlaceholder(
        ["verbalize", "authored_by"]
    )
    published_between: ConfigValuePlaceholder[Literal["Published between {start} and {end}."]] = ConfigValuePlaceholder(
        ["verbalize", "published_between"]
    )
    published_in_time: ConfigValuePlaceholder[Literal["Published in {start}."]] = ConfigValuePlaceholder(
        ["verbalize", "published_in_time"]
    )
    published_after: ConfigValuePlaceholder[Literal["Published after {start}."]] = ConfigValuePlaceholder(
        ["verbalize", "published_after"]
    )
    published_before: ConfigValuePlaceholder[Literal["Published before {end}."]] = ConfigValuePlaceholder(
        ["verbalize", "published_before"]
    )
    published_in_venues: ConfigValuePlaceholder[Literal["Published in {venues}."]] = ConfigValuePlaceholder(
        ["verbalize", "published_in_venues"]
    )
    content_must_satisfy: ConfigValuePlaceholder[
        Literal["To be considered relevant, the paper's content must satisfy:"]
    ] = ConfigValuePlaceholder(["verbalize", "content_must_satisfy"])
    criterion_line: ConfigValuePlaceholder[Literal["* **{name}**: {description}"]] = ConfigValuePlaceholder(
        ["verbalize", "criterion_line"]
    )
    nice_to_have: ConfigValuePlaceholder[Literal["It would be nice if the paper's content satisfies:"]] = (
        ConfigValuePlaceholder(["verbalize", "nice_to_have"])
    )
    specific_paper: ConfigValuePlaceholder[Literal["Look for a specific paper."]] = ConfigValuePlaceholder(
        ["verbalize", "specific_paper"]
    )
    set_of_papers: ConfigValuePlaceholder[Literal["Look for a set of papers."]] = ConfigValuePlaceholder(
        ["verbalize", "set_of_papers"]
    )
    bullet: ConfigValuePlaceholder[Literal["* {item}"]] = ConfigValuePlaceholder(["verbalize", "bullet"])
    metadata_criteria_heading: ConfigValuePlaceholder[Literal["**Metadata criteria:**"]] = ConfigValuePlaceholder(
        ["verbalize", "metadata_criteria_heading"]
    )
    content_criteria_heading: ConfigValuePlaceholder[Literal["**Content criteria:**"]] = ConfigValuePlaceholder(
        ["verbalize", "content_criteria_heading"]
    )
    search_for: ConfigValuePlaceholder[Literal['Search for "{content}"']] = ConfigValuePlaceholder(
        ["verbalize", "search_for"]
    )
    this_is_how: ConfigValuePlaceholder[Literal["This is how I interpreted your request:"]] = ConfigValuePlaceholder(
        ["verbalize", "this_is_how"]
    )


@dataclass(frozen=True)
class Sorting:
    explanation: Explanation = Explanation()


@dataclass(frozen=True)
class Explain:
    metadata: Metadata = Metadata()
    content: Content = Content()
    content_suffix: ContentSuffix = ContentSuffix()
    domains: Domains = Domains()
    summary: Summary = Summary()
    merged_summary: MergedSummary = MergedSummary()


@dataclass(frozen=True)
class ResponseTexts:
    metadata_agent: MetadataAgent = MetadataAgent()
    paper_finder_agent: PaperFinderAgent = PaperFinderAgent()
    refusal: Refusal = Refusal()
    detailed_response_prefix: ConfigValuePlaceholder[Literal["This is what I searched for:"]] = ConfigValuePlaceholder(
        ["response_texts", "detailed_response_prefix"]
    )
    suggest_broad_followup: ConfigValuePlaceholder[Literal["You can either refine your query or start a new one."]] = (
        ConfigValuePlaceholder(["response_texts", "suggest_broad_followup"])
    )
    suggest_to_search_for_papers_about_topic: ConfigValuePlaceholder[
        Literal["Do you want me to search for papers about this topic instead?"]
    ] = ConfigValuePlaceholder(["response_texts", "suggest_to_search_for_papers_about_topic"])
    work_harder_text: ConfigValuePlaceholder[
        Literal['You can ask me to "**work harder**" to run a more exhaustive search.']
    ] = ConfigValuePlaceholder(["response_texts", "work_harder_text"])
    generic_refusal_message: ConfigValuePlaceholder[
        Literal[
            "I'm sorry, I can't currently search for these criteria, or maybe I didn't understand what you meant. I work best for searching papers based on their content. Can you please rephrase your request?"
        ]
    ] = ConfigValuePlaceholder(["response_texts", "generic_refusal_message"])
    error_response_text: ConfigValuePlaceholder[
        Literal["I'm sorry, I had some technical issues and could not fulfill your request."]
    ] = ConfigValuePlaceholder(["response_texts", "error_response_text"])
    heavy_load_text: ConfigValuePlaceholder[
        Literal["I'm sorry, I'm experiencing heavy load right now, please try again later."]
    ] = ConfigValuePlaceholder(["response_texts", "heavy_load_text"])
    an_error_occurred: ConfigValuePlaceholder[Literal["An error occurred while processing your request: {message}"]] = (
        ConfigValuePlaceholder(["response_texts", "an_error_occurred"])
    )
    chitchat_reply: ConfigValuePlaceholder[Literal["Sorry, I don't know how to respond to that."]] = (
        ConfigValuePlaceholder(["response_texts", "chitchat_reply"])
    )


@dataclass(frozen=True)
class StepProgress:
    conversation_agent: ConversationAgent = ConversationAgent()
    round_executor: RoundExecutor = RoundExecutor()
    gather_user_needs: GatherUserNeeds = GatherUserNeeds()
    search_by_authors: SearchByAuthors = SearchByAuthors()
    llm_suggest: LlmSuggest = LlmSuggest()
    broad_by_specific_paper_citation: BroadBySpecificPaperCitation = BroadBySpecificPaperCitation()
    broad_search_by_keyword: BroadSearchByKeyword = BroadSearchByKeyword()
    fast_broad_search: FastBroadSearch = FastBroadSearch()
    broad_search: BroadSearch = BroadSearch()
    dense: Dense = Dense()
    metadata_only: MetadataOnly = MetadataOnly()
    specific_paper_by_name: SpecificPaperByName = SpecificPaperByName()
    specific_paper_by_title: SpecificPaperByTitle = SpecificPaperByTitle()
    paper_finder: PaperFinder = PaperFinder()
    snowball: Snowball = Snowball()


@dataclass(frozen=True)
class UserFacingSchema:
    step_progress: StepProgress = StepProgress()
    response_texts: ResponseTexts = ResponseTexts()
    explain: Explain = Explain()
    sorting: Sorting = Sorting()
    verbalize: Verbalize = Verbalize()

    def __getattr__(self, name: str) -> ConfigValuePlaceholder[str]:
        return ConfigValuePlaceholder([name])


uf = UserFacingSchema()
