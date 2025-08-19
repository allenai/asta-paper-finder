import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from ai2i.chain import LLMEndpoint, LLMModel, Timeouts, define_llm_endpoint
from ai2i.config import ConfigValue, config_value, configurable, ufv
from ai2i.dcollection import (
    RelevanceCriteria,
)

from mabool.agents.common.domain_utils import get_system_domain_params
from mabool.agents.query_analyzer.query_analyzer_prompts import (
    NameExtractionInput,
    decompose_query,
    name_extraction,
    specification_extraction,
    suitable_for_citing,
)
from mabool.agents.specific_paper_by_title.specific_paper_by_title_agent import get_specific_paper_by_title
from mabool.data_model.agent import (
    AnalyzedQuery,
    ConflictingOptionsError,
    ExtractedFields,
    ExtractedProperties,
    ExtractedYearlyTimeRange,
    MatchedTitle,
    NoActionableDataError,
    PartiallyAnalyzedQuery,
    QueryAnalysisFailure,
    QueryAnalysisPartialSuccess,
    QueryAnalysisRefusal,
    QueryAnalysisResult,
    QueryAnalysisSuccess,
    QueryAnalyzerError,
    QueryType,
)
from mabool.data_model.config import cfg_schema
from mabool.data_model.specifications import Specifications
from mabool.data_model.ufs import uf
from mabool.utils.llm_utils import get_api_key_for_model

logger = logging.getLogger(__name__)


def get_default_endpoint() -> LLMEndpoint:
    llm_model = LLMModel.from_name(config_value(cfg_schema.query_analyzer_agent.llm_abstraction_model_name))
    return define_llm_endpoint(
        default_timeout=Timeouts.medium,
        default_model=llm_model,
        logger=logger,
        api_key=get_api_key_for_model(llm_model),
    )


def extract_recency_and_centrality(
    fields: ExtractedFields, time_range: ExtractedYearlyTimeRange
) -> tuple[ExtractedProperties, list[QueryAnalyzerError]]:
    errors: list[QueryAnalyzerError] = []
    recent_first = fields["recency"].recency == "first"
    recent_last = fields["recency"].recency == "last"
    central_first = fields["centrality"].centrality == "first"
    central_last = fields["centrality"].centrality == "last"
    current_year = datetime.now().year

    if recent_first and recent_last:
        errors.append(ConflictingOptionsError("recent"))
    if central_first and central_last:
        errors.append(ConflictingOptionsError("central"))

    # in case max time is in the last year or so (or when minimum year is set while no limit for maximum year),
    #   lets sort by recency
    if (time_range.end is not None and time_range.end >= current_year - 1) or (
        time_range.end is None and time_range.start is not None
    ):
        recent_first = True

    extracted_properties = ExtractedProperties(
        recent_first=recent_first,
        recent_last=recent_last,
        central_first=central_first,
        central_last=central_last,
    )
    return extracted_properties, errors


def extract_time_range(fields: ExtractedFields) -> tuple[ExtractedYearlyTimeRange, list[QueryAnalyzerError]]:
    time_range = fields["time_range"]
    # Time Range extrema
    max_time_range = 0
    min_time_range = math.inf
    if time_range.start:
        min_time_range = min(time_range.start, min_time_range)
    if time_range.end:
        max_time_range = max(time_range.end, max_time_range)

    # we shouldnt have more than one min/max time constraints in a query but if we do lets take the global min and max
    #   then we can store them directly in the extracted_properties field
    if min_time_range != math.inf or max_time_range != 0:
        time_range = ExtractedYearlyTimeRange(
            start=int(min_time_range) if min_time_range != math.inf else None,
            end=max_time_range if max_time_range != 0 else None,
        )
    return time_range, []


@dataclass
class AnalyzeOutput:
    analyzed_query: AnalyzedQuery | PartiallyAnalyzedQuery
    specifications: Specifications
    errors: list[QueryAnalyzerError]


async def analyze(
    user_input: str, fields: ExtractedFields, specifications: Specifications, endpoint: LLMEndpoint
) -> AnalyzeOutput:
    errors: list[QueryAnalyzerError] = []

    extracted_content = fields["content"].content or ""
    relevance_criteria = fields["relevance_criteria"]
    authors = fields["authors"].authors
    venues = fields["venues"].venues
    domains = fields["domains"]
    possible_refusal = fields["possible_refusal"]

    failure_to_identify_criteria = False

    if extracted_content and not relevance_criteria.required_relevance_critieria:
        logger.warning("No relevance criteria, but content was extracted. Using content as single relevance criterion.")
        relevance_criteria = RelevanceCriteria.to_default_content_criteria(relevance_criteria, extracted_content)
        failure_to_identify_criteria = True

    time_range, es = extract_time_range(fields)
    errors.extend(es)

    extracted_properties, es = extract_recency_and_centrality(fields, time_range)
    errors.extend(es)

    # Broad vs. Specific
    query_type = None
    broad_or_specific = fields["broad_or_specific"].type
    matched_title = MatchedTitle(matched_title=None, matched_corpus_ids=None)
    if len(authors) > 0:
        # NOTE:
        #   For broad searches we prefer using by_author as it much better narrower, that is,
        #       with the exception for vespa which allows to search using authors as well.
        #   For specific papers, there are cases where there is no content
        #       or the specific paper is neither by name or title but instead with author+description,
        #   Thus, for now, if we see authors we route to the by_author agent
        query_type = QueryType(type="BY_AUTHOR", broad_or_specific=broad_or_specific)
    elif not extracted_content:
        if len(venues) > 0 or time_range.non_empty():
            # this agent actually solves venue/year queries.
            query_type = QueryType(type="METADATA_ONLY_NO_AUTHOR", broad_or_specific=broad_or_specific)
        else:
            errors.append(NoActionableDataError(ufv(uf.response_texts.generic_refusal_message)))
    # NOTE: currently "papers about" is how the conversation manager signals that this is a modified query.
    elif broad_or_specific == "broad" or user_input.lower().startswith("papers about"):
        query_type = QueryType(type="BROAD_BY_DESCRIPTION", broad_or_specific="broad")
    else:
        by_name_or_title = fields["by_name_or_title"].type
        if by_name_or_title == "name":
            query_type = QueryType(type="SPECIFIC_BY_NAME", broad_or_specific="specific")
        else:
            query_type = QueryType(type="SPECIFIC_BY_TITLE", broad_or_specific="specific")
            candidates, extracted_title = await get_specific_paper_by_title(user_input, time_range, venues, authors)
            if extracted_title and len(candidates.documents) > 0:
                matched_title = MatchedTitle(
                    matched_title=extracted_title, matched_corpus_ids=[doc.corpus_id for doc in candidates.documents]
                )
            else:
                query_type = QueryType(type="BROAD_BY_DESCRIPTION", broad_or_specific="broad")

    should_extract_name = False
    if query_type and query_type.type == "BROAD_BY_DESCRIPTION":
        if len(extracted_content.split()) > 3 and not failure_to_identify_criteria:
            should_extract_name = True
    if query_type and (query_type.type == "SPECIFIC_BY_NAME" or query_type.type == "SPECIFIC_BY_TITLE"):
        should_extract_name = True

    if should_extract_name:
        try:
            endpoint = endpoint.timeout(Timeouts.short)
            extracted_name = await endpoint.execute(name_extraction).once(
                NameExtractionInput(query=user_input, **get_system_domain_params(domains))
            )
        except Exception as e:
            logger.warning(f"Failed to extract name from query: {user_input}. Error: {e}")
            extracted_name = None
        if extracted_name:
            extracted_properties.specific_paper_name = extracted_name

    if query_type and query_type.type == "SPECIFIC_BY_NAME" and not extracted_properties.specific_paper_name:
        logger.warning("Specific by name but no name extracted, no actionable data.")
        errors.append(NoActionableDataError(ufv(uf.response_texts.generic_refusal_message)))
    if query_type and query_type.type == "BROAD_BY_DESCRIPTION" and extracted_properties.specific_paper_name:
        try:
            endpoint = endpoint.model_params(temperature=0.0).timeout(Timeouts.short)
            suitable_for_by_citing = await endpoint.execute(suitable_for_citing).once(
                {"query": user_input, "extracted_name": extracted_properties.specific_paper_name}
            )
        except Exception as e:
            logger.exception(
                f"Failed to classify if query is suitable for by citing strategy: {user_input}. Error: {e}"
            )
            suitable_for_by_citing = False
        extracted_properties.suitable_for_by_citing = suitable_for_by_citing
    if not errors and query_type is not None:
        return AnalyzeOutput(
            analyzed_query=AnalyzedQuery(
                original_query=user_input,
                content=extracted_content,
                authors=authors,
                venues=venues,
                time_range=time_range,
                extracted_properties=extracted_properties,
                query_type=query_type,
                relevance_criteria=relevance_criteria,
                domains=domains,
                possible_refusal=possible_refusal,
                matched_title=matched_title,
            ),
            specifications=specifications,
            errors=[],
        )
    else:
        return AnalyzeOutput(
            analyzed_query=PartiallyAnalyzedQuery(
                original_query=user_input,
                content=extracted_content,
                authors=authors,
                venues=venues,
                time_range=time_range,
                extracted_properties=extracted_properties,
                query_type=query_type,
                relevance_criteria=relevance_criteria,
                domains=domains,
                possible_refusal=possible_refusal,
                matched_title=matched_title,
            ),
            specifications=specifications,
            errors=errors,
        )


@configurable
async def extract_specifications(
    user_input: str,
    specification_extraction_model_name: str = ConfigValue(cfg_schema.metadata_planner_agent.llm_model_name),
) -> Specifications:
    try:
        spec_extract_llm_model = LLMModel.from_name(specification_extraction_model_name)
        spec_extract_endpoint = define_llm_endpoint(
            default_timeout=Timeouts.medium,
            default_model=spec_extract_llm_model,
            logger=logger,
            api_key=get_api_key_for_model(spec_extract_llm_model),
        ).model_params(temperature=0.1)
        return await spec_extract_endpoint.execute(specification_extraction).once(user_input)
    except Exception as e:
        logger.exception(f"Failed to extract specifications: {user_input}. Continue with best effort. Error: {e}")
        return Specifications(union=[])


@configurable
async def decompose_and_analyze_query(
    user_input: str,
    query_analysis_model_name: str = ConfigValue(cfg_schema.query_analyzer_agent.llm_abstraction_model_name),
) -> AnalyzeOutput:
    analysis_llm_model = LLMModel.from_name(query_analysis_model_name)
    analysis_endpoint = define_llm_endpoint(
        default_timeout=Timeouts.medium,
        default_model=analysis_llm_model,
        logger=logger,
        api_key=get_api_key_for_model(analysis_llm_model),
    ).model_params(temperature=0.0)

    tasks = (analysis_endpoint.execute(decompose_query).once(user_input), extract_specifications(user_input))
    extracted_fields, specifications = await asyncio.gather(*tasks)
    if config_value(cfg_schema.query_analyzer_agent.force_broad):
        extracted_fields["broad_or_specific"].type = "broad"
        extracted_fields["authors"].authors = []

    analyze_output = await analyze(user_input, extracted_fields, specifications, analysis_endpoint)
    return analyze_output


@configurable
async def decompose_and_analyze_query_restricted(
    user_input: str,
    model_name: str = ConfigValue(cfg_schema.query_analyzer_agent.llm_abstraction_model_name),
) -> QueryAnalysisResult:
    analyze_output = await decompose_and_analyze_query(user_input, model_name)
    analyzed_query = analyze_output.analyzed_query
    errors = analyze_output.errors
    match analyzed_query:
        case AnalyzedQuery():
            logger.info(f"{analyzed_query = }")
        case PartiallyAnalyzedQuery():
            logger.warning(f"{analyzed_query = } with {errors = }")

    if any(isinstance(error, NoActionableDataError) for error in errors):
        return QueryAnalysisFailure(
            partially_analyzed_query=cast(PartiallyAnalyzedQuery, analyzed_query),
            error=[e for e in errors if isinstance(e, NoActionableDataError)][0],
        )

    if analyzed_query.possible_refusal.type:
        return QueryAnalysisRefusal(analysis=analyzed_query, errors=errors)

    if len(errors) == 0:
        return QueryAnalysisSuccess(
            analyzed_query=cast(AnalyzedQuery, analyzed_query), specifications=analyze_output.specifications
        )

    return QueryAnalysisPartialSuccess(
        partially_analyzed_query=cast(PartiallyAnalyzedQuery, analyzed_query), errors=errors
    )
