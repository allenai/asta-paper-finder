import logging
import re
from datetime import date, datetime, timedelta
from typing import Sequence, cast

from semanticscholar.Author import Author
from semanticscholar.Citation import Citation
from semanticscholar.PaginatedResults import PaginatedResults
from semanticscholar.Paper import Paper
from semanticscholar.SemanticScholarException import ObjectNotFoundException

from ai2i.common.utils.asyncio import custom_gather
from ai2i.dcollection import PaperFinderDocument
from ai2i.dcollection.data_access_context import DocumentCollectionContext
from ai2i.dcollection.external_api.s2.author import (
    AUTHOR_PAPERS_PER_CALL_LIMIT,
)
from ai2i.dcollection.external_api.s2.common import AsyncPaginatedResults, s2_retry
from ai2i.dcollection.interface.collection import Document
from ai2i.dcollection.interface.document import (
    ExtractedYearlyTimeRange,
    OriginQuery,
    S2CitingPapersQuery,
    S2PaperRelevanceSearchQuery,
    S2PaperTitleSearchQuery,
)
from ai2i.dcollection.loaders.s2_rest import s2_paper_to_document

logger = logging.getLogger(__name__)

PAPER_SEARCH_PER_CALL_LIMIT = 100
PAPER_SEARCH_TOTAL_LIMIT = 1000
TOTAL_AUTHORS_LIMIT = 9999
MAX_PAPERS_BY_AUTHORS_NAME = 100


def merge_time_range_and_inserted_before(
    time_range: ExtractedYearlyTimeRange | None, inserted_before: str | None
) -> str | None:
    start = ""
    if time_range and time_range.start:
        start = time_range.start

    # check which end date is sooner
    last_allowed_pub_date = get_publication_date_from_inserted_before(inserted_before)
    end = ""
    if last_allowed_pub_date:
        if time_range and time_range.end and time_range.end < last_allowed_pub_date.year:
            end_year = time_range.end
            end = time_range.end
        else:
            end_year = last_allowed_pub_date.year
            end = str(last_allowed_pub_date)
        if start and start > end_year:
            return None

    return f"{start}:{end}" if (start or end) else ""


def get_publication_date_from_inserted_before(inserted_before: str | None) -> date | None:
    res = None

    if inserted_before:
        hyphened_parts_count = len(inserted_before.split("-"))
        final_format = "-".join("%Y-%m-%d".split("-")[:hyphened_parts_count])
        inclusive = datetime.strptime(inserted_before, final_format) - timedelta(days=1)
        res = inclusive.date()

    return res


def check_if_paper_inserted_before(
    last_allowed_pub_date: date | None, year: int | None, publication_date: date | None
) -> bool:
    return bool(
        not last_allowed_pub_date
        or (
            (not publication_date or publication_date <= last_allowed_pub_date)
            and year
            and (year <= last_allowed_pub_date.year)
        )
    )


def get_by_title_origin_query(
    query: str,
    time_range: ExtractedYearlyTimeRange | None = None,
    venues: Sequence[str] | None = None,
) -> OriginQuery:
    return OriginQuery(
        query_type="s2_title_search",
        query=S2PaperTitleSearchQuery(query=query, time_range=time_range, venues=venues),
        ranks=[1],
    )


@s2_retry()
async def s2_papers_by_title(
    query: str,
    context: DocumentCollectionContext,
    time_range: ExtractedYearlyTimeRange | None = None,
    venues: Sequence[str] | None = None,
    inserted_before: str | None = None,
) -> Sequence[Document]:
    clean_query = re.sub(r"[:^]", "", query).replace("-", " ")
    publication_date_or_year = merge_time_range_and_inserted_before(time_range, inserted_before)
    if publication_date_or_year is None:
        # this means start date is larger than end date
        return []
    try:
        paper = cast(
            Paper,
            await context.s2_client.search_paper(
                query=clean_query,
                fields=["corpusId"],
                limit=1,
                venue=cast(list[str], venues or []),
                match_title=True,
                publication_date_or_year=publication_date_or_year,
            ),
        )
        return [
            s2_paper_to_document(
                corpus_id=paper.corpusId,
                paper=paper,
                origin_query=get_by_title_origin_query(query, time_range, venues),
            )
        ]
    except Exception as e:
        logger.warning(f"Failed to fetch paper by title: {query}, error: {e}")
        if isinstance(e, ObjectNotFoundException):
            return []
        raise e


@s2_retry()
async def s2_paper_search(
    query: str,
    search_iteration: int,
    context: DocumentCollectionContext,
    limit: int = PAPER_SEARCH_PER_CALL_LIMIT,
    time_range: ExtractedYearlyTimeRange | None = None,
    venues: Sequence[str] | None = None,
    fields_of_study: list[str] | None = None,
    min_citations: int | None = None,
    total_limit: int = PAPER_SEARCH_TOTAL_LIMIT,
    inserted_before: str | None = None,
) -> Sequence[Document]:
    clean_query = re.sub(r"[:^]", "", query).replace("-", " ")
    bulk = True if clean_query == "" else False
    publication_date_or_year = merge_time_range_and_inserted_before(time_range, inserted_before)
    if publication_date_or_year is None:
        # this means start date is larger than end date
        return []
    search_paper_result = await context.s2_client.search_paper(
        query=clean_query,
        fields=["corpusId"],
        limit=min(limit, PAPER_SEARCH_PER_CALL_LIMIT, total_limit),
        venue=cast(list[str], venues or []),
        fields_of_study=fields_of_study if fields_of_study else [],
        min_citation_count=min_citations or 0,
        bulk=bulk,
        publication_date_or_year=publication_date_or_year,
    )

    s2_papers = [
        paper
        async for paper in AsyncPaginatedResults[Paper](
            results=cast(PaginatedResults, search_paper_result), max_results=total_limit
        )
    ]
    return [
        s2_paper_to_document(
            corpus_id=paper.corpusId,
            paper=paper,
            origin_query=OriginQuery(
                query_type="s2_bulk_search" if bulk else "s2_relevance_search",
                query=S2PaperRelevanceSearchQuery(query=query, num_results=limit, time_range=time_range, venues=venues),
                iteration=search_iteration,
                ranks=[i + 1],
            ),
        )
        for i, paper in enumerate(s2_papers)
    ]


async def s2_by_author(
    author_profiles: Sequence[Author],
    context: DocumentCollectionContext,
    inserted_before: str | None = None,
) -> Sequence[Document]:
    s2docs: list[Document] = []
    if not author_profiles:
        return s2docs
    else:
        s2docs: list[Document] = [
            doc
            for docs in await custom_gather(
                *[
                    s2_papers_by_author_id(author.authorId, context=context, inserted_before=inserted_before)
                    for author in author_profiles
                ],
                force_deterministic=context.force_deterministic,
            )
            for doc in docs
        ]
    return s2docs


@s2_retry()
async def s2_papers_by_author_id(
    author_id: str,
    context: DocumentCollectionContext,
    inserted_before: str | None = None,
) -> Sequence[Document]:
    author_papers_result = await context.s2_client.get_author_papers(
        author_id=author_id, fields=["corpusId", "year", "publicationDate"], limit=AUTHOR_PAPERS_PER_CALL_LIMIT
    )
    last_allowed_pub_date = get_publication_date_from_inserted_before(inserted_before)

    return [
        PaperFinderDocument(corpus_id=str(paper.corpusId))
        async for paper in AsyncPaginatedResults[Paper](results=author_papers_result)
        if check_if_paper_inserted_before(
            last_allowed_pub_date, paper.year, paper.publicationDate.date() if paper.publicationDate else None
        )
    ]


@s2_retry()
async def s2_fetch_citing_papers(
    corpus_id: str,
    search_iteration: int,
    context: DocumentCollectionContext,
    total_limit: int = 1000,
    inserted_before: str | None = None,
) -> Sequence[Document]:
    fields = ["corpusId", "title", "year", "contexts", "publicationDate"]
    citing_papers_result = await context.s2_client.get_paper_citations(f"CorpusId:{corpus_id}", fields)
    s2_papers = [
        citation async for citation in AsyncPaginatedResults[Citation](citing_papers_result, max_results=total_limit)
    ]
    last_allowed_pub_date = get_publication_date_from_inserted_before(inserted_before)
    date_filtered_papers = [
        citation
        for citation in s2_papers
        if check_if_paper_inserted_before(
            last_allowed_pub_date,
            citation.paper.year,
            citation.paper.publicationDate.date() if citation.paper.publicationDate else None,
        )
    ]

    return [
        s2_paper_to_document(
            corpus_id=citation.paper.corpusId,
            paper=citation.paper,
            origin_query=OriginQuery(
                query_type="s2_citing_papers",
                query=S2CitingPapersQuery(corpus_id=corpus_id),
                iteration=search_iteration,
                ranks=[i + 1],
            ),
            contexts=citation.contexts,
        )
        for i, citation in enumerate(date_filtered_papers)
    ]
