from __future__ import annotations

import json
import logging
import math
from functools import partial
from typing import Literal, TypedDict

from ai2i.dcollection import Document, DocumentCollection, DocumentFieldName, ExtractedYearlyTimeRange
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


ControlType = Literal["failure", "control", "result"]


class AgentState(BaseModel):
    checkpoint: DocumentCollection | None = None

    @field_validator("checkpoint", mode="before")
    @classmethod
    def copy_doc_collection(cls, doc_collection: DocumentCollection | None) -> DocumentCollection | None:
        if doc_collection:
            copied_documents = [d.model_copy(deep=True) for d in doc_collection.documents]
            return doc_collection.factory.from_docs(copied_documents)
        return doc_collection


def filter_by_time_range_with_buffer(
    docs: DocumentCollection,
    time_range: ExtractedYearlyTimeRange,
    keep_missing: bool = True,
    use_buffer: bool = False,
) -> DocumentCollection:
    if not time_range.start and not time_range.end:
        return docs
    buffer = 0
    if time_range.start and time_range.end and use_buffer:
        buffer = math.ceil((time_range.end - time_range.start + 1) * (20 / 100))  # +/-20%

    if buffer:
        logger.info(f"Adding buffer of +/- {buffer} years to the time range")

    def _is_doc_in_timerange_with_buffer(doc: Document) -> bool:
        if not doc.year:
            if keep_missing:
                logger.warning("filter_by_time_range_with_buffer: Keeping the doc even though year is not set")
                return True
            return False

        start = time_range.start - buffer if time_range.start else 0
        end = time_range.end + buffer if time_range.end else math.inf

        if start <= doc.year <= end:
            return True

        return False

    return docs.filter(_is_doc_in_timerange_with_buffer)


def filter_by_venues(doc: Document, venues: list[str], keep_missing: bool = False) -> bool:
    if not (doc.publication_venue or doc.venue):
        if keep_missing:
            logger.warning("filter_by_venues: Keeping the doc even though both venue and publication_venue are not set")
            return True
        return False

    found_names_lowered_set = set()
    if doc.venue:
        found_names_lowered_set.add(doc.venue.lower())
    if doc.publication_venue:
        found_names_lowered_set.add(doc.publication_venue.normalized_name)
        if doc.publication_venue.alternate_names:
            found_names_lowered_set.update([v.lower() for v in doc.publication_venue.alternate_names])
    requested_venues_lowered_set = set(v.lower() for v in venues)

    # if one of the requested venues appears as one of the alternative names then it's a match
    if found_names_lowered_set.intersection(requested_venues_lowered_set):
        return True
    return False


def filter_by_author(expected_authors: list[str], doc: Document, keep_missing: bool | None = False) -> bool:
    if not expected_authors:
        return True

    if not doc.authors:
        if keep_missing:
            # lets not throw away the doc if it doesnt have its authors set
            logger.warning("filter_by_author: Keeping the doc even though authors are not set")
            return True
        return False

    for expected_author in expected_authors:
        expected_name_parts = expected_author.lower().split()
        expected_initials = (
            (expected_name_parts[0][0], expected_name_parts[-1][0]) if len(expected_name_parts) > 1 else ()
        )
        matched = False
        for found_author in doc.authors:
            found_name_parts = found_author.name.lower().split()
            found_initials = (found_name_parts[0][0], found_name_parts[-1][0]) if len(found_name_parts) > 1 else ()
            if expected_name_parts[-1] == found_name_parts[-1] and (
                not expected_initials or not found_initials or expected_initials == found_initials
            ):
                matched = True
                break
        if not matched:
            return False
    return True


async def filter_docs_by_metadata(
    docs: DocumentCollection,
    time_range: ExtractedYearlyTimeRange | None = None,
    venues: list[str] | None = None,
    authors: list[str] | None = None,
    keep_missing: bool = False,
    use_time_buffer: bool = False,
) -> DocumentCollection:
    fields_to_load: list[DocumentFieldName] = []
    if time_range and time_range.non_empty():
        fields_to_load.append("year")
    if venues:
        fields_to_load.extend(["venue", "publication_venue"])
    if authors:
        fields_to_load.append("authors")

    if not fields_to_load:
        return docs
    else:
        logger.info(f"Number of documents before filter: {len(docs)}")
        docs = await docs.with_fields(fields_to_load)

    if time_range and time_range.non_empty():
        logger.info(f"Filtering documents by time range: {time_range.start} <= year <= {time_range.end}")
        docs = filter_by_time_range_with_buffer(docs, time_range, keep_missing, use_time_buffer)

    if venues:
        logger.info(f"Filtering documents by venue list: {', '.join(venues)}")
        filter_by_venues_partial = partial(filter_by_venues, venues=venues, keep_missing=keep_missing)
        docs = docs.filter(filter_by_venues_partial)

    if authors:
        logger.info(f"Filtering documents by author list: {', '.join(authors)}")
        filter_by_author_partial = partial(filter_by_author, expected_authors=authors, keep_missing=keep_missing)
        docs = docs.filter(filter_by_author_partial)

    logger.info(f"Number of documents after filter: {len(docs)}")

    return docs


class InputQuery(TypedDict):
    query: str


class InputQueryJson(InputQuery):
    query_json: str


def as_input_query(query: str) -> InputQuery:
    return {"query": query}


def as_input_query_json(query: str) -> InputQueryJson:
    return {"query_json": json.dumps({"query": query}), "query": query}
