import logging

from ai2i.config import config_value
from ai2i.dcollection import DocumentCollection, rj_4l_codes
from ai2i.di import DI

from mabool.agents.common.step_reporting_utils import verbalize_list
from mabool.data_model.config import cfg_schema

logger = logging.getLogger(__name__)

NUM_WORKERS = 100
MAX_DOCS = 5000

DOC_ID_FIELDS = ["url", "title", "year", "corpus_id", "abstract"]
TRUNCATE_ABSTRACT_CHARS = 1500


def verbalize_relavance_counts(documents: DocumentCollection, drop_zeros: bool = False) -> str:
    """
    Example output:
      41 perfectly relevant, 21 highly relevant and 0 somewhat relevant
    """
    docs_by_relevance = sorted(
        documents.group_by(
            lambda doc: doc.relevance_judgement.relevance if doc.relevance_judgement is not None else -1
        ).items(),
        key=lambda k: k[0],
        reverse=True,
    )
    if drop_zeros:
        docs_by_relevance = [(rel, docs) for rel, docs in docs_by_relevance if len(docs) > 0]
    codes_to_labels = {value: key for key, value in rj_4l_codes.items()}
    relevance_count_strings = [
        f"{len(docs)} {codes_to_labels[relevance]}" for relevance, docs in docs_by_relevance if relevance >= 1
    ]
    return verbalize_list(relevance_count_strings, "and")


@DI.managed
async def report_relevance_judgement_counts(documents: DocumentCollection) -> None:
    # Since we report found relevance papers during the process, and the total counts in the final user messaeg, we
    # disable this global report for now.
    return
    # await step_progress_reporter.report_step(f"Found {verbalize_relavance_counts(documents, drop_zeros=True)} papers.")


async def log_relevance_value_counts(
    documents: DocumentCollection,
) -> None:
    logger.info("Relevance judgement results:")
    logger.info(verbalize_relavance_counts(documents))


def get_relevant_docs(documents: DocumentCollection, threshold: int | None = None) -> DocumentCollection:
    if threshold is None:
        if config_value(cfg_schema.relevance_judgement.keep_irrelevant_docs):
            threshold = 0
        else:
            threshold = 1

    return documents.filter(
        lambda doc: doc.relevance_judgement is not None and doc.relevance_judgement.relevance >= threshold
    )


def count_relevant_docs(documents: DocumentCollection, threshold: int = 1) -> int:
    return len(get_relevant_docs(documents, threshold))
