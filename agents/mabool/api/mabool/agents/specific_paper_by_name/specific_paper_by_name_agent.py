import logging
import re
from collections import Counter, defaultdict
from typing import Awaitable, Optional, Sequence

from ai2i.config import ConfigValue, config_value, configurable
from ai2i.dcollection import (
    AggTransformComputedField,
    AssignedField,
    DocLoadingError,
    Document,
    DocumentCollection,
    DocumentCollectionSortDef,
    ExtractedYearlyTimeRange,
    S2PaperRelevanceSearchQuery,
)
from ai2i.di import DI

from mabool.agents.common.common import (
    AgentState,
)
from mabool.agents.common.domain_utils import get_fields_of_study_filter_from_domains
from mabool.agents.common.utils import alog_args
from mabool.agents.llm_suggestion.llm_suggestion_agent import get_llm_suggested_papers
from mabool.data_model.agent import (
    AgentInput,
    AgentOutput,
    DomainsIdentified,
)
from mabool.data_model.config import cfg_schema
from mabool.infra.operatives import CompleteResponse, Operative, OperativeResponse
from mabool.utils.asyncio import custom_gather
from mabool.utils.dc import DC

logger = logging.getLogger(__name__)


class SpecificPaperByNameInput(AgentInput):
    user_input: str
    extracted_name: str
    extracted_content: Optional[str] = None
    time_range: Optional[ExtractedYearlyTimeRange] = None
    venues: Optional[list[str]] = None
    authors: Optional[list[str]] = None
    domains: DomainsIdentified


type SpecificPaperByNameState = AgentState
type SpecificPaperByNameOutput = AgentOutput


async def get_top_cited_corpus_ids(
    docs: DocumentCollection,
) -> Counter[str]:
    cited_corpus_ids_counter: Counter[str] = Counter()
    for doc in docs.documents:
        if not doc.snippets:
            continue
        for snippet in doc.snippets:
            if snippet.ref_mentions:
                cited_corpus_ids_counter.update([rm.matched_paper_corpus_id for rm in snippet.ref_mentions])
    return cited_corpus_ids_counter


def get_top_candidate_by_count(cited_corpus_ids_counter: Counter[str]) -> Counter[str] | None:
    logger.info(f"Top cited corpus_ids: {cited_corpus_ids_counter.most_common(5)}")
    if len(cited_corpus_ids_counter) == 0:
        logger.warning("No cited corpus_ids found in the results")
        return None
    if len(cited_corpus_ids_counter) == 1:
        logger.info(
            f"Only one cited cited corpus_id found in the results: {cited_corpus_ids_counter.most_common(1)[0]}"
        )
        corpus_ids_counter = cited_corpus_ids_counter
    else:
        logger.info(f"Found {len(cited_corpus_ids_counter)} unique corpus_ids")
        first_place = cited_corpus_ids_counter.most_common(1)[0]
        second_place = cited_corpus_ids_counter.most_common(2)[1]
        # if the first place has more than 1.5x the citations of the second place, we consider it a good signal
        if first_place[1] >= 1.5 * second_place[1]:
            logger.info(
                f"Top cited corpus_ids: {first_place[0]} with {first_place[1]} citations, "
                f"{first_place[1] / second_place[1]:.1f} times more than second place."
            )
            corpus_ids_counter = Counter({first_place[0]: first_place[1]})
        else:
            logger.warning(
                f"Top cited corpus_ids: {first_place[0]} with {first_place[1]} citations, "
                f"only {first_place[1] / second_place[1]:.1f} times more than second place. Not a strong signal."
            )
            return None

    return corpus_ids_counter


def capitalized_weirdly(name: str) -> bool:
    # not all lower and not all words with first letter capitalized
    return not name.islower() and not name.replace("-", "").istitle()


def we_introduce(name: str, abstract: str) -> bool:
    # check if the abstract contains a sentence that introduces the name
    # e.g. "We introduce a new model called X...", "Our model, X is..."
    match = re.search(
        r"((we|this work|this paper) ([^\s\.]+\s+){0,15}?(present|introduce|propose|publish|design|develop)(s|es|ed)? "
        rf"([^\s\.]+\s+){{0,15}}(\( ?)?({re.escape(name)})(\( ?)?)|((Our ([^\s]+\s+)"
        rf"{{0,7}}( ?called|named)?({re.escape(name)})))",
        abstract,
        re.IGNORECASE,
    )
    return bool(match)


def url_contains_name(name: str, abstract: str) -> bool:
    # match a link for name in the abstract (case-insensitive)
    # must be an exact match, not just a substring
    # e.g. for the name "spike", the following are good matches:
    # last part of the path: "http://github.com/allenai/spike" or "https://github.io/spike/"
    # first part of the dns: "http://spike.github.io" or "https://spike.allenai.org"
    # dns part: "https://spike.ai" or https://spike.org"
    # this is bad match:  "https://spikeallenai.org" or "https://allen.ai/spikey"
    # currently also catches stuff like "https://allen.ai/vision-spike" and "https://allen.ai/spike-vision"
    # which may give false positives
    match = re.search(rf"(https?://)[^\s]*\b{re.escape(name)}\b", abstract, re.IGNORECASE)
    if match:
        return True

    return False


def title_introduces_name(name: str, title: str) -> bool:
    valid_name_re = rf"([Tt]he )?{re.escape(name)}([ \-]?[vV]?(\d(\.\d)?)?)"
    begin_match = re.match(rf"^{valid_name_re}: ", title)
    if begin_match:
        return True
    end_match = re.search(rf"\({valid_name_re}\)\s*$", title)
    if end_match:
        return True
    return False


def score_paper_for_name(
    extracted_name: str,
    doc: Document,
) -> float:
    def scoring_func(
        extracted_name: str,
        found_in_search_with_content: int,
        title: str,
        abstract: str,
        strong_multiplier: float = 0.5,
        weak_multiplier: float = 0.25,
    ) -> float:
        # strong_multiplier is used for things that are strong signals of relevance, so it is larger than
        # weak_multiplier which is used for weaker signals
        score = 0.0

        if title_introduces_name(extracted_name, title):
            score += 1.5 * strong_multiplier
        elif extracted_name in title:
            score += 0.5 * weak_multiplier
        if abstract:
            num_occurrences = len(re.findall(rf"\b{re.escape(extracted_name)}\b", abstract))
            score += min(num_occurrences, 3) * 0.1 * strong_multiplier
            if num_occurrences > 0:
                if we_introduce(extracted_name, abstract):
                    score += 1 * strong_multiplier

        if url_contains_name(extracted_name, abstract):
            score += 1

        if found_in_search_with_content:
            score *= 2

        return score

    found_in_search_with_content = doc.origins is not None and any(
        isinstance(origin.query, S2PaperRelevanceSearchQuery) and len(origin.query.query) > len(extracted_name)
        for origin in doc.origins
    )
    title = doc.title or ""
    abstract = doc.abstract or ""
    case_insensitive_score = scoring_func(
        extracted_name.lower(),
        found_in_search_with_content,
        title.lower(),
        abstract.lower(),
        strong_multiplier=0.5,
        weak_multiplier=0.2,
    )
    if capitalized_weirdly(extracted_name):
        case_sensitive_score = scoring_func(
            extracted_name, found_in_search_with_content, title, abstract, strong_multiplier=1.0, weak_multiplier=1
        )
    elif extracted_name.lower() == extracted_name:
        case_sensitive_score = scoring_func(
            extracted_name, found_in_search_with_content, title, abstract, strong_multiplier=0.75, weak_multiplier=0.5
        )
    else:
        case_sensitive_score = case_insensitive_score
    return max(case_insensitive_score, case_sensitive_score)


async def s2_name_relevance_search(
    user_input: str,
    extracted_name: str,
    domains: DomainsIdentified,
    extracted_content: Optional[str],
    time_range: Optional[ExtractedYearlyTimeRange],
    venues: Optional[list[str]],
    search_iteration: int = 1,
) -> DocumentCollection:
    fields_of_study = get_fields_of_study_filter_from_domains(domains)

    # search for the extracted name in S2, score it based on the name and the content and filter out low scores
    s2_futures: list[Awaitable[DocumentCollection]] = []
    s2_futures.append(
        DC.from_s2_search(
            query=extracted_name, limit=5, time_range=time_range, venues=venues, fields_of_study=fields_of_study
        )
    )
    stripped_content = (
        " ".join(
            extracted_content.lower()
            .replace(extracted_name.lower(), "")
            .replace("paper", "")
            .replace("the", "")
            .replace("original", "")
            .strip()
            .split()
        )
        if extracted_content
        else ""
    )
    if stripped_content:
        s2_futures.append(
            DC.from_s2_search(
                query=extracted_name + " " + stripped_content,
                limit=5,
                time_range=time_range,
                venues=venues,
                fields_of_study=fields_of_study,
                search_iteration=search_iteration,
            )
        )
    s2_results: Sequence[DocumentCollection] = await custom_gather(*s2_futures)
    s2_results_merged = s2_results[0].merged(*s2_results[1:])

    docs_with_scores = await s2_results_merged.with_fields(
        [
            AssignedField[float](
                field_name="s2_search_score",
                assigned_values=[score_paper_for_name(extracted_name, doc) for doc in s2_results_merged.documents],
            ),
        ]
    )

    sorted_docs = docs_with_scores.sorted(
        sort_definitions=[DocumentCollectionSortDef(field_name="s2_search_score", order="desc")]
    )

    def _score_above(doc: Document) -> bool:
        score = doc.dynamic_value("s2_search_score", float, default=0.0)
        return score is not None and score >= 0.5

    # keep only those with score >= 0.5
    filtered_docs = sorted_docs.filter(_score_above)
    return filtered_docs


async def score_specific_by_name_all_origins(
    merged_results: DocumentCollection,
    llm_suggest_results: DocumentCollection,
    s2_search_results: DocumentCollection,
    spike_most_cited_results: DocumentCollection,
) -> DocumentCollection:
    def score_by_rank(collection: DocumentCollection, max_score: int = 3) -> dict[str, float]:
        rank_scores = {doc.corpus_id: min(len(collection) - i, max_score) for i, doc in enumerate(collection.documents)}

        def normalize_scores(scores: dict[str, int]) -> dict[str, float]:
            if not scores:
                return {}
            max_score = max(scores.values())
            min_score = min(scores.values())
            if max_score == min_score:
                return {corpus_id: 1 for corpus_id in scores.keys()}
            return {corpus_id: (score - min_score) / (max_score - min_score) for corpus_id, score in scores.items()}

        normalized_rank_scores = normalize_scores(rank_scores)

        # the base score for just being in the results is 1, the rank is added to it as a bonus
        return {corpus_id: score + 1 for corpus_id, score in normalized_rank_scores.items()}

    logger.info("Scoring results")
    llm_suggest_scores = score_by_rank(llm_suggest_results)
    s2_search_scores = score_by_rank(s2_search_results)
    spike_most_cited_scores = score_by_rank(spike_most_cited_results)

    def weighted_scoring_func(scoring_maps: Sequence[dict[str, float]], weights: Sequence[float]) -> dict[str, float]:
        if len(scoring_maps) != len(weights):
            raise ValueError("scoring_maps and weights must have the same length")
        scores: dict[str, float] = defaultdict(float)
        for scoring_map, weight in zip(scoring_maps, weights):
            for corpus_id, score in scoring_map.items():
                scores[corpus_id] += score * weight
        return scores

    weighted_scores = weighted_scoring_func([llm_suggest_scores, spike_most_cited_scores, s2_search_scores], [1, 1, 1])

    results_with_scores = await merged_results.with_fields(
        [
            AssignedField[float](
                field_name="specific_paper_by_name_score",
                assigned_values=[weighted_scores.get(doc.corpus_id, 0) for doc in merged_results.documents],
            ),
        ]
    )
    return results_with_scores


@DI.managed
async def get_specific_paper_by_name_with_reporting(
    user_input: str,
    extracted_name: str,
    domains: DomainsIdentified,
    extracted_content: str | None = None,
    time_range: ExtractedYearlyTimeRange | None = None,
    venues: list[str] | None = None,
    authors: list[str] | None = None,
    filter_threshold: float | None = None,
    search_iteration: int = 1,
) -> DocumentCollection:
    if extracted_content is None:
        return DC.empty()
    if filter_threshold is None:
        filter_threshold = 0.0

    llm_suggest_future = get_llm_suggested_papers(
        user_input=user_input,
        domains=domains,
        extra_hints=(
            "The query may refer to a method name, technique name, dataset name, model name, "
            "or any other name mentioned in the query, if so, suggest the paper that is most relevant to that resource."
        ),
        search_iteration=search_iteration,
    )

    s2_search_future = s2_name_relevance_search(
        user_input,
        extracted_name,
        domains,
        extracted_content,
        time_range,
        venues,
        search_iteration=search_iteration,
    )

    logger.info("Gathering results from all sources")
    llm_suggest_results, s2_search_results = await custom_gather(
        llm_suggest_future, s2_search_future, return_exceptions=True
    )

    if llm_suggest_results is None or isinstance(llm_suggest_results, BaseException):
        logger.warning(f"Error while fetching LLM suggestions: {llm_suggest_results}")
        llm_suggest_results = DC.empty()
    if s2_search_results is None or isinstance(s2_search_results, BaseException):
        logger.warning(f"Error while fetching S2 search results: {s2_search_results}")
        s2_search_results = DC.empty()
    # if spike_most_cited_results is None or isinstance(spike_most_cited_results, BaseException):
    #     logger.warning(f"Error while fetching SPIKE most cited results: {spike_most_cited_results}")
    #     spike_most_cited_results = DC.empty()

    logger.info("All results gathered, merging...")
    merged_results = s2_search_results + llm_suggest_results  # + spike_most_cited_results
    logger.info(f"Merged results: {len(merged_results.documents or [])} candidate documents")

    results_with_scores = await score_specific_by_name_all_origins(
        merged_results=merged_results,
        llm_suggest_results=llm_suggest_results,
        s2_search_results=s2_search_results,
        spike_most_cited_results=DC.empty(),  # spike_most_cited_results,
    )

    if len(results_with_scores) > 1:
        results_with_scores = results_with_scores.sorted(
            sort_definitions=[DocumentCollectionSortDef(field_name="specific_paper_by_name_score", order="desc")]
        )

    # filter out results with score lower than filter_threshold
    if filter_threshold:
        logger.info(f"Filtering out results with score lower than {filter_threshold}")
        results_with_scores = results_with_scores.filter(
            lambda doc: doc.specific_paper_by_name_score > filter_threshold  # type: ignore
        )

    if not results_with_scores.documents:
        logger.warning("No results found after filtering")
        return DC.empty()

    logger.info(f"Results after filtering: {len(results_with_scores.documents)} documents")

    # add final_agent_score batch computed field as normalized specific_paper_by_name_score
    async def calculate_final_specific_paper_by_name_score(
        docs: Sequence[Document],
    ) -> Sequence[float | DocLoadingError]:
        if not docs or not any(doc["specific_paper_by_name_score"] is not None for doc in docs):
            return [
                DocLoadingError(
                    corpus_id=d.corpus_id, original_exception=ValueError("missing specific_paper_by_name_score")
                )
                for d in docs
            ]
        max_score = max(
            doc["specific_paper_by_name_score"] for doc in docs if doc["specific_paper_by_name_score"] is not None
        )
        min_score = min(
            0,
            min(doc["specific_paper_by_name_score"] for doc in docs if doc["specific_paper_by_name_score"] is not None),
        )
        if max_score == min_score:
            return [1.0] * len(docs)
        return [((doc["specific_paper_by_name_score"] or 0) - min_score) / (max_score - min_score) for doc in docs]

    results_with_scores = await results_with_scores.with_fields(
        [
            AggTransformComputedField[float](
                field_name="final_specific_paper_by_name_score",
                computation_func=calculate_final_specific_paper_by_name_score,
                required_fields=["specific_paper_by_name_score"],
            ),
        ]
    )
    return results_with_scores


@configurable
async def get_specific_paper_by_name(
    user_input: str,
    extracted_name: str,
    domains: DomainsIdentified,
    extracted_content: str | None = None,
    time_range: ExtractedYearlyTimeRange | None = None,
    venues: list[str] | None = None,
    authors: list[str] | None = None,
    filter_threshold: float | None = ConfigValue(cfg_schema.specific_paper_by_name_agent.filter_threshold),
    search_iteration: int = 1,
) -> DocumentCollection:
    return await get_specific_paper_by_name_with_reporting(
        user_input,
        extracted_name,
        domains,
        extracted_content,
        time_range,
        venues,
        authors,
        filter_threshold,
        search_iteration,
    )


class SpecificPaperByNameAgent(
    Operative[SpecificPaperByNameInput, SpecificPaperByNameOutput, SpecificPaperByNameState]
):
    def register(self) -> None: ...

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self, state: SpecificPaperByNameState | None, inputs: SpecificPaperByNameInput
    ) -> tuple[SpecificPaperByNameState | None, OperativeResponse[SpecificPaperByNameOutput]]:
        search_results = await get_specific_paper_by_name(
            user_input=inputs.user_input,
            extracted_name=inputs.extracted_name,
            extracted_content=inputs.extracted_content,
            time_range=inputs.time_range,
            venues=inputs.venues,
            authors=inputs.authors,
            domains=inputs.domains,
            filter_threshold=config_value(cfg_schema.specific_paper_by_name_agent.filter_threshold),
        )

        top2_results = search_results.take(config_value(cfg_schema.search_by_author_agent.limit_for_specific))

        return (
            state,
            CompleteResponse(data=AgentOutput(response_text="", doc_collection=top2_results)),
        )
