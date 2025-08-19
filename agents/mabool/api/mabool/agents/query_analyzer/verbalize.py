from ai2i.config import ufv

from mabool.agents.common.step_reporting_utils import verbalize_list
from mabool.data_model.agent import ExtractedYearlyTimeRange, QueryAnalysisResult, RelevanceCriteria, get_analyzed_query
from mabool.data_model.ufs import uf


def verbalize_authors(authors: list[str]) -> str | None:
    if len(authors) > 0:
        return ufv(uf.verbalize.authored_by, authors=verbalize_list(authors, "and"))
    return None


def verbalize_time_range(time_range: ExtractedYearlyTimeRange) -> str | None:
    if time_range.is_empty():
        return None
    if time_range.start and time_range.end and time_range.start != time_range.end:
        return ufv(uf.verbalize.published_between, start=time_range.start, end=time_range.end)
    if time_range.start and time_range.end and time_range.start == time_range.end:
        return ufv(uf.verbalize.published_in_time, start=time_range.start)
    if time_range.start and not time_range.end:
        return ufv(uf.verbalize.published_after, start=time_range.start)
    if time_range.end and not time_range.start:
        return ufv(uf.verbalize.published_before, end=time_range.end)
    return None


def verbalize_venues(venues: list[str]) -> str | None:
    if len(venues) == 0:
        return None
    return ufv(uf.verbalize.published_in_venues, venues=verbalize_list(venues, "or"))


def verbalize_relevance_criteria(criteria: RelevanceCriteria, show_nice_to_have: bool = False) -> str | None:
    required = criteria.required_relevance_critieria or []
    nice_to_have = criteria.nice_to_have_relevance_criteria or []
    response: list[str] = []
    if len(required) > 0:
        response.append(f"\n{ufv(uf.verbalize.content_must_satisfy)}\n\n")
        for c in required:
            response.append(ufv(uf.verbalize.criterion_line, name=c.name, description=c.description))
    if len(nice_to_have) > 0 and show_nice_to_have:
        response.append(f"\n\n{ufv(uf.verbalize.nice_to_have)}")
        for c in nice_to_have:
            response.append(ufv(uf.verbalize.criterion_line, name=c.name, description=c.description))
    if len(response) == 0:
        return None
    return "\n".join(response)


def verbalize_analyzed_query(query_analysis_result: QueryAnalysisResult) -> str | None:
    analyzed_query = get_analyzed_query(query_analysis_result)

    request_str: list[str] = []

    # Broad or specific
    is_broad = False
    if analyzed_query.query_type is not None:
        if analyzed_query.query_type.broad_or_specific == "specific":
            request_str.append(ufv(uf.verbalize.specific_paper))
            if analyzed_query.extracted_properties.specific_paper_name:
                request_str.append(
                    ufv(uf.verbalize.bullet, item=analyzed_query.extracted_properties.specific_paper_name)
                )
        if analyzed_query.query_type.broad_or_specific == "broad":
            is_broad = True
            request_str.append(ufv(uf.verbalize.set_of_papers))

    metadata_criteria = []
    if len(analyzed_query.authors) > 0:
        metadata_criteria.append(verbalize_authors(analyzed_query.authors))
    if not analyzed_query.time_range.is_empty():
        metadata_criteria.append(verbalize_time_range(analyzed_query.time_range))
    if len(analyzed_query.venues) > 0:
        metadata_criteria.append(verbalize_venues(analyzed_query.venues))

    if len(metadata_criteria) > 0:
        request_str.append(f"\n{ufv(uf.verbalize.metadata_criteria_heading)}")
        for item in metadata_criteria:
            request_str.append(ufv(uf.verbalize.bullet, item=item))

    if analyzed_query.content != "" and is_broad:
        request_str.append(f"\n{ufv(uf.verbalize.content_criteria_heading)}")
        request_str.append(ufv(uf.verbalize.search_for, content=analyzed_query.content))
        request_str.append("")
        relevance = verbalize_relevance_criteria(analyzed_query.relevance_criteria)
        if relevance is not None:
            request_str.append(relevance)

    if len(request_str) == 0:
        return None
    return "\n".join([f"{ufv(uf.verbalize.this_is_how)}\n"] + request_str)
