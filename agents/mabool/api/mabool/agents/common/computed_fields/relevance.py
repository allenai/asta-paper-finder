from __future__ import annotations

import json
import logging
from functools import partial
from typing import Any, Sequence

from ai2i.chain import (
    ChainComputation,
    LLMModel,
    ResponseMetadata,
    Timeouts,
    define_chat_llm_call,
    define_llm_endpoint,
    system_message,
    user_message,
)
from ai2i.config import config_value
from ai2i.dcollection import (
    BatchComputedField,
    CorpusId,
    DocLoadingError,
    Document,
    RelevanceCriteria,
    RelevanceCriterion,
    RelevanceCriterionJudgement,
    RelevanceJudgement,
)
from ai2i.di import DI
from pydantic import BaseModel, create_model
from tenacity import stop_after_attempt

from mabool.agents.common.computed_fields.relevance_types import (
    RELEVANCE_LABEL_TO_SCORE,
    DocumentRelevanceInput,
    LLMJudgementResult,
    RelevanceCriterionJudgementValue,
    RelevanceScores,
    RelevanceThresholds,
)
from mabool.agents.common.computed_fields.relevant_snippets import find_relevant_snippet
from mabool.data_model.config import cfg_schema
from mabool.utils import tracing_deps
from mabool.utils.llm_utils import get_api_key_for_model
from mabool.utils.metrics import Metrics

logger = logging.getLogger(__name__)


relevance_criteria_judgement_prompt_with_relevant_snippets_after = """
Judge how relevant the following paper is to each of the provided criteria. For each criterion, consider its entire description when making your judgement.

For each criterion, provide the following outputs:
- `relevance` (str): one of "Perfectly Relevant", "Somewhat Relevant", "Not Relevant".
- `relevant_snippet` (str | null): a snippet from the document that best show the relevance of the paper to the criterion. To be clear, copy EXACT text ONLY. Choose one short text span that best shows the relevance in a concrete and specific way, up to 20 words. ONLY IF NECESSARY, you can add another few-words-long span (e.g. for coreference, disambiguation, necessary context), separated by ` ... `. If relevance is "Not Relevant" output null. The snippet may contain citations, but make sure to only take snippets that directly show the relevance of this paper. 

Also provide another field `relevance_summary` (str): a short summary explanation of how the paper is relevant to the criteria in general. null if it is not relevant to any of the criteria.
- This should be short but convey the most useful and specific information for a user skimming a list of papers, up to 30 words.
- No need to mention which are perfectly relevant, somewhat relevant, or not relevant. Just provide new information that was not mentioned in the criteria.
- Start with perfectly relevant ones, include a specific and interesting detail about what matches them. Then go on to somewhat relevant ones, saying why it is close but not a perfect match. No need to add extra info for not relevant ones.
- Start with actionable info. Instead of saying "The paper uses X to...", just say "Uses X to...".

Output a JSON:
- top-level key `criteria`. Under it, for every criterion name (exactly as given in the provided criteria), there should be an object containing two fields: `relevance` and `relevant_snippet`.
- top-level key `relevance_summary` with string value or null.

Criteria:
```
{{{criteria}}}
```"""


def relevance_judgement_field(
    relevance_criteria: RelevanceCriteria, field_name: str = "relevance_judgement"
) -> BatchComputedField[RelevanceJudgement]:
    return BatchComputedField[RelevanceJudgement](
        field_name=field_name,
        computation_func=partial(_load_relevance_judgement, relevance_criteria=relevance_criteria),
        required_fields=["markdown", "title", "abstract", "snippets", "citation_contexts"],
    )


@DI.managed
async def _load_relevance_judgement(
    entities: Sequence[Document],
    relevance_criteria: RelevanceCriteria,
    metrics: Metrics = DI.requires(tracing_deps.metrics),
) -> Sequence[RelevanceJudgement | DocLoadingError]:
    if not entities:
        logger.info("No valid documents to run relevance judgement on.")
        return []

    llm_results = await _judge_documents_relevance(entities, relevance_criteria)

    judgements_by_id, failed_docs_errors = _process_llm_results_with_failures(
        llm_results, entities, relevance_criteria, metrics
    )

    ordered_results = _to_original_entities_order(entities, judgements_by_id, failed_docs_errors)

    return ordered_results


async def _judge_documents_relevance(
    documents: Sequence[Document],
    relevance_criteria: RelevanceCriteria,
) -> list[LLMJudgementResult | Exception]:
    criteria_list = relevance_criteria.to_flat_criteria(include_nice_to_have=False)
    if not criteria_list:
        raise Exception("No valid relevance criteria available.")

    document_inputs = _prepare_documents_for_llm(documents, criteria_list)
    if not document_inputs:
        logger.info("No documents to run relevance judgement on.")
        return []

    logger.info(f"Running relevance judgement for {len(document_inputs)} documents")

    judge_relevance_chain = _create_relevance_judgement_chain(criteria_list)
    return await _execute_llm_judgements(document_inputs, judge_relevance_chain)


def _prepare_documents_for_llm(
    documents: Sequence[Document], criteria: list[RelevanceCriterion]
) -> list[DocumentRelevanceInput]:
    return [
        {
            "document": doc["markdown"],
            "criteria": json.dumps(
                [{"name": criterion.name, "description": criterion.description} for criterion in criteria], indent=2
            ),
            "doc_id": doc["corpus_id"],
        }
        for doc in documents
    ]


def _create_relevance_judgement_chain(criteria: list[RelevanceCriterion]) -> ChainComputation:
    relevance_result_type = _create_dynamic_judgement_result_type(criteria)

    judge_relevance = (
        define_chat_llm_call(
            [
                system_message(relevance_criteria_judgement_prompt_with_relevant_snippets_after),
                user_message("{{document}}"),
            ],
            format="mustache",
            input_type=DocumentRelevanceInput,
            output_type=relevance_result_type,
            include_response_metadata=True,
        )
        .passthrough_input()
        .map(_extract_llm_response)
    )
    return judge_relevance


def _create_dynamic_judgement_result_type(criteria: list[RelevanceCriterion]) -> type[BaseModel]:
    fields: dict[str, Any] = {criterion.name: (RelevanceCriterionJudgementValue, ...) for criterion in criteria}
    criteria_model = create_model("CriteriaResult", **fields)

    return create_model(
        "RelevanceJudgementResult",
        **{
            "criteria": (criteria_model, ...),
            "relevance_summary": (str | None, ...),
        },
    )


def _extract_llm_response(
    input_and_response: tuple[DocumentRelevanceInput, tuple[BaseModel, ResponseMetadata]],
) -> LLMJudgementResult:
    input_data, (response_model, metadata) = input_and_response
    model_name = metadata.get("model_name") or metadata.get("model_version") or "unspecified model"

    return {
        "doc_id": input_data["doc_id"],
        "model_name": model_name,
        "criteria_judgements": response_model.model_dump(),
    }


async def _execute_llm_judgements(
    inputs: list[DocumentRelevanceInput], judge_relevance_chain: ChainComputation
) -> list[LLMJudgementResult | Exception]:
    llm_model = LLMModel.from_name(config_value(cfg_schema.relevance_judgement.relevance_model_name), temperature=0.0)
    endpoint = define_llm_endpoint(
        default_timeout=Timeouts.medium,
        default_model=llm_model,
        stop=stop_after_attempt(3),
        api_key=get_api_key_for_model(llm_model),
    )

    return await endpoint.execute(judge_relevance_chain).many(
        inputs,
        return_exceptions=True,
        max_concurrency=config_value(cfg_schema.relevance_judgement.openai_concurrency),
    )


def _process_llm_results_with_failures(
    llm_results: list[LLMJudgementResult | Exception],
    original_documents: Sequence[Document],
    relevance_criteria: RelevanceCriteria,
    metrics: Metrics,
) -> tuple[dict[CorpusId, RelevanceJudgement], dict[CorpusId, DocLoadingError]]:
    successful_results = []
    failed_docs_errors = dict[CorpusId, DocLoadingError]()

    for i, result in enumerate(llm_results):
        if isinstance(result, Exception):
            failed_doc_id = original_documents[i].corpus_id
            failed_docs_errors[failed_doc_id] = _create_doc_loading_error(
                failed_doc_id, reason="", original_exception=result
            )
            metrics.relevance_judgement_failures += 1
            logger.warning("Relevance judgement failed. Skipping.")
        else:
            successful_results.append(result)

    scores = _convert_to_relevance_scores(successful_results, metrics)
    judgements_by_id = {}

    for doc in original_documents:
        if doc.corpus_id not in failed_docs_errors.keys():
            judgement_data = scores["judgements_by_doc"].get(doc.corpus_id)
            if judgement_data:
                judgement_or_error = _try_create_document_judgement(
                    doc, scores, judgement_data, relevance_criteria, metrics
                )
                if isinstance(judgement_or_error, RelevanceJudgement):
                    judgements_by_id[doc.corpus_id] = judgement_or_error
                else:
                    failed_docs_errors[doc.corpus_id] = judgement_or_error

    return judgements_by_id, failed_docs_errors


def _convert_to_relevance_scores(results: list[LLMJudgementResult], metrics: Metrics) -> RelevanceScores:
    judgements_by_doc: dict[CorpusId, list[dict[str, Any]]] = {}
    summaries_by_doc: dict[CorpusId, str | None] = {}
    models_by_doc: dict[CorpusId, str] = {}

    for result in results:
        doc_id = result["doc_id"]
        criteria_data = result["criteria_judgements"]

        doc_judgements = [
            {
                "name": criterion_name,
                "relevance": RELEVANCE_LABEL_TO_SCORE[judgement["relevance"]],
                "relevant_snippet": judgement["relevant_snippet"],
            }
            for criterion_name, judgement in criteria_data["criteria"].items()
        ]

        judgements_by_doc[doc_id] = doc_judgements
        summaries_by_doc[doc_id] = criteria_data["relevance_summary"]
        models_by_doc[doc_id] = result["model_name"]

        metrics.add_relevance_judged(doc_id)

    return {
        "judgements_by_doc": judgements_by_doc,
        "summaries_by_doc": summaries_by_doc,
        "models_by_doc": models_by_doc,
    }


def _to_original_entities_order(
    entities: Sequence[Document],
    judgements_by_id: dict[CorpusId, RelevanceJudgement],
    failed_doc_errors: dict[CorpusId, DocLoadingError],
) -> list[RelevanceJudgement | DocLoadingError]:
    results = []

    for entity in entities:
        if entity.corpus_id in failed_doc_errors.keys():
            results.append(failed_doc_errors[entity.corpus_id])
        else:
            judgement = judgements_by_id.get(entity.corpus_id)
            assert judgement is not None, (
                f"Document {entity.corpus_id} not found in judgements_by_id but also not in failed_doc_ids"
            )
            results.append(judgement)

    return results


def _try_create_document_judgement(
    doc: Document,
    scores: RelevanceScores,
    judgement_data: list[dict[str, Any]],
    relevance_criteria: RelevanceCriteria,
    metrics: Metrics,
) -> RelevanceJudgement | DocLoadingError:
    doc_id = doc.corpus_id

    try:
        criterion_judgements = _build_relevance_criteria_judgements(doc, judgement_data)
        if criterion_judgements is None:
            metrics.relevance_judgement_failures += 1
            return _create_doc_loading_error(doc_id, "Failed to build criterion judgements")

        validation_error = _validate_required_criteria(doc_id, criterion_judgements, relevance_criteria, metrics)
        if validation_error:
            return validation_error

        relevance_score = _compute_document_relevance_score(doc_id, criterion_judgements, relevance_criteria, metrics)
        if isinstance(relevance_score, DocLoadingError):
            return relevance_score

        return _build_relevance_judgement(doc, scores, criterion_judgements, relevance_score)

    except Exception as e:
        metrics.relevance_judgement_failures += 1
        logger.exception(f"Failed to create judgement for {doc_id}: {e}")
        return _create_doc_loading_error(doc_id, f"Unexpected error: {e}", e)


def _validate_required_criteria(
    doc_id: CorpusId,
    criterion_judgements: list[RelevanceCriterionJudgement],
    relevance_criteria: RelevanceCriteria,
    metrics: Metrics,
) -> DocLoadingError | None:
    """Validate that all required criteria are present in judgements."""
    if relevance_criteria is None:
        return None

    required_criteria_names = [
        criterion.name for criterion in relevance_criteria.to_flat_criteria(include_nice_to_have=False)
    ]
    judgement_criteria_names = [criterion.name for criterion in criterion_judgements]

    if any(name not in judgement_criteria_names for name in required_criteria_names):
        metrics.relevance_judgement_failures += 1
        logger.warning(
            f"Required relevance criteria not found for document {doc_id}. Skipping. "
            f"Required criteria: {required_criteria_names}. Judged criteria: {judgement_criteria_names}"
        )
        return _create_doc_loading_error(doc_id, "Required relevance criteria not found")

    return None


def _compute_document_relevance_score(
    doc_id: CorpusId,
    criterion_judgements: list[RelevanceCriterionJudgement],
    relevance_criteria: RelevanceCriteria,
    metrics: Metrics,
) -> float | DocLoadingError:
    try:
        return _calculate_relevance_criteria_score(relevance_criteria, criterion_judgements)
    except Exception as e:
        metrics.relevance_judgement_failures += 1
        logger.exception(f"Failed to calculate relevance criteria score for document {doc_id}. Skipping. {e}")
        return _create_doc_loading_error(doc_id, f"Failed to calculate relevance score: {e}", e)


def _build_relevance_judgement(
    doc: Document,
    scores: RelevanceScores,
    criterion_judgements: list[RelevanceCriterionJudgement],
    relevance_score: float,
) -> RelevanceJudgement:
    """Build the final RelevanceJudgement object."""
    doc_id = doc.corpus_id
    relevance_level = _convert_relevance_score_to_level(relevance_score)

    return RelevanceJudgement(
        relevance=relevance_level,
        relevance_model_name=scores["models_by_doc"].get(
            doc_id, config_value(cfg_schema.relevance_judgement.relevance_model_name)
        ),
        relevance_criteria_judgements=criterion_judgements,
        relevance_score=relevance_score,
        relevance_summary=scores["summaries_by_doc"].get(doc_id),
    )


def _create_doc_loading_error(
    doc_id: CorpusId, reason: str, original_exception: Exception | None = None
) -> DocLoadingError:
    return DocLoadingError(corpus_id=doc_id, original_exception=original_exception or Exception(reason))


def _build_relevance_criteria_judgements(
    doc: Document, judgement_data: list[dict[str, Any]]
) -> list[RelevanceCriterionJudgement] | None:
    try:
        judgements = []
        for criterion in judgement_data:
            try:
                relevant_snippets = find_relevant_snippet(doc, criterion["relevant_snippet"])
            except Exception as e:
                logger.exception(f"Failed to find relevant snippet: {e}")
                relevant_snippets = None

            judgements.append(
                RelevanceCriterionJudgement(
                    name=criterion["name"],
                    relevance=criterion["relevance"],
                    relevant_snippets=relevant_snippets,
                )
            )
        return judgements
    except Exception:
        logger.exception("Failed to build relevance criteria judgements")
        return None


def _calculate_relevance_criteria_score(
    relevance_criteria: RelevanceCriteria, judgements: list[RelevanceCriterionJudgement]
) -> float:
    """Calculate weighted relevance score based on criteria weights (RESTORED from original)."""
    criterion_name_to_weight = {}
    for criteria in relevance_criteria.to_flat_criteria():
        criterion_name_to_weight[criteria.name] = criteria.weight

    score = 0
    for judgement in judgements:
        score += criterion_name_to_weight[judgement.name] * judgement.relevance / 3

    score = min(1.0, score)
    return score


def _convert_relevance_score_to_level(score: float) -> int:
    """Convert relevance score to 0-3 level (UPDATED to use new threshold constants)."""
    if score <= RelevanceThresholds.NOT_RELEVANT:
        return 0
    elif score <= RelevanceThresholds.SOMEWHAT_RELEVANT:
        return 1
    elif score <= RelevanceThresholds.HIGHLY_RELEVANT:
        return 2
    return 3


async def load_relevance_judgement_v1(
    entities: Sequence[Document], relevance_criteria: RelevanceCriteria
) -> Sequence[RelevanceJudgement | DocLoadingError]:
    return await _load_relevance_judgement(entities, relevance_criteria)
