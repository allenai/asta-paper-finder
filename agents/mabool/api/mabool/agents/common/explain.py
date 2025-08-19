from typing import Sequence, Tuple

from ai2i.config import ufv
from ai2i.dcollection import Document, DocumentCollection

from mabool.agents.common.relevance_judgement_utils import get_relevant_docs
from mabool.agents.common.step_reporting_utils import counted_noun, pluralize
from mabool.data_model.agent import AnalyzedQuery, PartiallyAnalyzedQuery
from mabool.data_model.ufs import uf
from mabool.utils.text import AND_CONNECTOR, COMMA_CONNECTOR, JOIN_SUFFIX, LIST_SEPARATOR


def explain_metadata(analyzed_input: AnalyzedQuery | PartiallyAnalyzedQuery) -> str:
    metadata_sections = []

    if analyzed_input.time_range.non_empty():
        if analyzed_input.time_range.start == analyzed_input.time_range.end:
            time_text = ufv(uf.explain.metadata.time_same_year, year=analyzed_input.time_range.start)
        else:
            if analyzed_input.time_range.start and analyzed_input.time_range.end:
                time_text = ufv(
                    uf.explain.metadata.time_range_start_and_end,
                    start=analyzed_input.time_range.start,
                    end=analyzed_input.time_range.end,
                )
            elif analyzed_input.time_range.start:
                time_text = ufv(uf.explain.metadata.time_range_start_only, start=analyzed_input.time_range.start)
            else:
                time_text = ufv(uf.explain.metadata.time_range_end_only, end=analyzed_input.time_range.end)

        metadata_sections.append(ufv(uf.explain.metadata.time_range_prefix, time=time_text))

    if analyzed_input.venues:
        venues_list = LIST_SEPARATOR.join(analyzed_input.venues)
        metadata_sections.append(
            ufv(uf.explain.metadata.venue, s="s" if len(analyzed_input.venues) > 1 else "", items=venues_list)
        )

    if analyzed_input.authors:
        authors_list = LIST_SEPARATOR.join(analyzed_input.authors)
        metadata_sections.append(
            ufv(uf.explain.metadata.author, s="s" if len(analyzed_input.authors) > 1 else "", items=authors_list)
        )

    return "\n".join(metadata_sections)


def explain_domains(analyzed_input: AnalyzedQuery | PartiallyAnalyzedQuery) -> str:
    if not analyzed_input.domains:
        return ""

    domain_sections = []

    domain_sections.append(ufv(uf.explain.domains.main_field, field=analyzed_input.domains.main_field))

    if analyzed_input.domains.key_secondary_fields:
        secondary_fields = LIST_SEPARATOR.join(analyzed_input.domains.key_secondary_fields)
        domain_sections.append(ufv(uf.explain.domains.secondary_fields, fields=secondary_fields))

    return "\n".join(domain_sections)


def explain_query_analysis(analyzed_input: AnalyzedQuery | PartiallyAnalyzedQuery) -> list[str]:
    metadata_explained = explain_metadata(analyzed_input)
    domains_explained = explain_domains(analyzed_input)
    domains_explained = None  # we do not currently take domains into account when searching.

    content_sections = []

    if analyzed_input.query_type and analyzed_input.query_type.type == "SPECIFIC_BY_NAME":
        name = analyzed_input.extracted_properties.specific_paper_name
        content_suffix = (
            ufv(uf.explain.content_suffix.with_content, content=analyzed_input.content)
            if analyzed_input.content
            else ""
        )
        content_sections.append(
            ufv(uf.explain.content.specific_paper_by_name, name=name, content_suffix=content_suffix)
        )
    # NOTE: we check here for SPECIFIC_BY_TITLE specifically as there is also the option of BY_AUTHOR + specific
    elif analyzed_input.query_type and analyzed_input.query_type.type == "SPECIFIC_BY_TITLE" and analyzed_input.content:
        content_sections.append(
            ufv(uf.explain.content.specific_paper_by_title, title=analyzed_input.matched_title.matched_title)
        )

    elif analyzed_input.content:
        content_sections.append(ufv(uf.explain.content.papers_discussing, content=analyzed_input.content))

        has_relevance_criteria = (
            analyzed_input.relevance_criteria
            and analyzed_input.relevance_criteria.required_relevance_critieria
            and not (
                analyzed_input.relevance_criteria.clarification_questions
                or analyzed_input.relevance_criteria.is_default()
            )
        )

        if has_relevance_criteria:
            content_sections.append(ufv(uf.explain.content.relevance_criteria_header))

            sorted_criteria = (
                sorted(
                    analyzed_input.relevance_criteria.required_relevance_critieria,
                    key=lambda x: x.weight,
                    reverse=True,
                )
                if analyzed_input.relevance_criteria.required_relevance_critieria
                else []
            )

            for criterion in sorted_criteria:
                content_sections.append(
                    ufv(uf.explain.content.criterion_format, name=criterion.name, description=criterion.description)
                )

    if domains_explained:
        content_sections.append(f"{ufv(uf.explain.content.fields_of_study_header)}\n{domains_explained}")

    if metadata_explained:
        content_sections.append(f"{ufv(uf.explain.content.metadata_header)}\n{metadata_explained}")

    return ["\n".join(content_sections)]


def _count_relevance_by_level(collection: DocumentCollection) -> Tuple[int, int, int]:
    perfect_count = len(
        collection.filter(lambda doc: doc.relevance_judgement is not None and doc.relevance_judgement.relevance == 3)
    )
    high_count = len(
        collection.filter(lambda doc: doc.relevance_judgement is not None and doc.relevance_judgement.relevance == 2)
    )
    other_count = len(
        collection.filter(lambda doc: doc.relevance_judgement is not None and doc.relevance_judgement.relevance == 1)
    )

    return perfect_count, high_count, other_count


def generate_response_summary(collection: DocumentCollection, is_broad: bool, has_content: bool) -> str:
    if is_broad and has_content:
        results = get_relevant_docs(collection).documents
    else:
        results = collection.documents

    if not results:
        return ufv(uf.explain.summary.no_results)

    perfect_count, high_count, other_count = _count_relevance_by_level(collection)

    if not has_content:
        return ufv(uf.explain.summary.found_papers, paper_count=counted_noun("paper", len(results)))

    if not is_broad:
        return ufv(uf.explain.summary.potential_matches, paper_count=counted_noun("paper", len(results)))

    if perfect_count > 0:
        return _build_response_with_perfect_matches(perfect_count, high_count, other_count)

    return _build_response_without_perfect_matches(high_count, other_count)


def _build_response_with_perfect_matches(perfect_count: int, high_count: int, other_count: int) -> str:
    parts = []

    if perfect_count > 1:
        parts.append(ufv(uf.explain.summary.perfect_matches.plural, count=perfect_count))
    else:
        parts.append(ufv(uf.explain.summary.perfect_matches.singular))

    if high_count > 0:
        connector = COMMA_CONNECTOR if other_count > 0 else AND_CONNECTOR
        if high_count > 1:
            parts.append(ufv(uf.explain.summary.highly_relevant.plural, connector=connector, count=high_count))
        else:
            parts.append(ufv(uf.explain.summary.highly_relevant.singular, connector=connector, count=high_count))

    if other_count > 0:
        parts.append(ufv(uf.explain.summary.other_relevant.default, others=counted_noun("other", other_count)))

    return "".join(parts) + JOIN_SUFFIX


def _build_response_without_perfect_matches(high_count: int, other_count: int) -> str:
    parts = [ufv(uf.explain.summary.no_perfect.intro)]

    if high_count > 0:
        if high_count == 1:
            parts.append(ufv(uf.explain.summary.no_perfect.highly_relevant_only_singular))
        else:
            parts.append(ufv(uf.explain.summary.no_perfect.highly_relevant_only_plural, count=high_count))

        if other_count > 0:
            parts.append(
                ufv(
                    uf.explain.summary.no_perfect.and_others,
                    count=other_count,
                    others_label=pluralize("other", other_count),
                )
            )
    else:
        if other_count == 1:
            parts.append(ufv(uf.explain.summary.no_perfect.potentially_relevant_singular))
        else:
            parts.append(ufv(uf.explain.summary.no_perfect.potentially_relevant_plural, count=other_count))

    return "".join(parts) + JOIN_SUFFIX


def generate_response_summary_for_merged_collection(
    prev_collection: DocumentCollection, current_collection: DocumentCollection, is_broad: bool, has_content: bool
) -> str:
    prev_perfect_count = len(
        prev_collection.filter(lambda doc: (rj := doc.relevance_judgement) is not None and rj.relevance == 3)
    )

    if prev_perfect_count > 0:
        if prev_perfect_count > 1:
            prev_summary = ufv(uf.explain.merged_summary.previous_search.perfect_plural, count=prev_perfect_count)
        else:
            prev_summary = ufv(uf.explain.merged_summary.previous_search.perfect_singular)
    else:
        prev_summary = ufv(uf.explain.merged_summary.previous_search.no_perfect)

    if is_broad and has_content:
        relevant_docs = get_relevant_docs(current_collection).documents
    else:
        relevant_docs = current_collection.documents

    if len(relevant_docs) == 0:
        return prev_summary + ufv(uf.explain.merged_summary.previous_search.no_additional)

    perfect_count, high_count, other_count = _count_relevance_by_level(current_collection)

    if not is_broad or not has_content:
        return _build_merged_response_potentially_relevant(prev_summary, relevant_docs)

    if perfect_count > 0:
        return _build_merged_response_with_perfect_matches(prev_summary, perfect_count, high_count, other_count)

    return _build_merged_response_without_perfect_matches(prev_summary, high_count, other_count)


def _build_merged_response_potentially_relevant(prev_summary: str, relevant_docs: Sequence[Document]) -> str:
    if len(relevant_docs) == 1:
        return prev_summary + ufv(uf.explain.merged_summary.potentially_relevant.singular)
    else:
        return prev_summary + ufv(uf.explain.merged_summary.potentially_relevant.plural, count=len(relevant_docs))


def _build_merged_response_with_perfect_matches(
    prev_summary: str, perfect_count: int, high_count: int, other_count: int
) -> str:
    perfect_intro = ufv(uf.explain.merged_summary.perfect_intro)
    parts = [prev_summary, perfect_intro]

    if perfect_count > 1:
        parts.append(ufv(uf.explain.merged_summary.perfect_matches.plural, count=perfect_count))
    else:
        parts.append(ufv(uf.explain.merged_summary.perfect_matches.singular))

    if high_count > 0:
        connector = COMMA_CONNECTOR if other_count > 0 else AND_CONNECTOR
        if high_count > 1:
            parts.append(ufv(uf.explain.summary.highly_relevant.plural, connector=connector, count=high_count))
        else:
            parts.append(ufv(uf.explain.summary.highly_relevant.singular, connector=connector, count=high_count))

    if other_count > 0:
        parts.append(ufv(uf.explain.summary.other_relevant.default, others=pluralize("other", other_count)))

    return "".join(parts) + JOIN_SUFFIX


def _build_merged_response_without_perfect_matches(prev_summary: str, high_count: int, other_count: int) -> str:
    parts = [ufv(uf.explain.merged_summary.no_perfect_matches.intro)]

    if high_count > 0:
        if high_count == 1:
            parts.append(ufv(uf.explain.merged_summary.no_perfect_matches.more_highly_relevant_singular))
        else:
            parts.append(
                ufv(uf.explain.merged_summary.no_perfect_matches.more_highly_relevant_plural, count=high_count)
            )

        if other_count > 0:
            parts.append(
                ufv(
                    uf.explain.summary.no_perfect.and_others,
                    count=other_count,
                    others_label=pluralize("other", other_count),
                )
            )
    else:
        if other_count == 1:
            parts.append(ufv(uf.explain.merged_summary.no_perfect_matches.more_potentially_relevant_singular))
        else:
            parts.append(
                ufv(uf.explain.merged_summary.no_perfect_matches.more_potentially_relevant_plural, count=other_count)
            )

    return prev_summary + " " + "".join(parts) + JOIN_SUFFIX
