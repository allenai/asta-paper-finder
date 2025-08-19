from __future__ import annotations

import logging
from typing import Literal, Mapping, Sequence, cast

from inflection import camelize
from semanticscholar.AsyncSemanticScholar import AsyncSemanticScholar
from semanticscholar.Paper import Paper
from semanticscholar.SemanticScholarException import BadQueryParametersException, NoMorePagesException

from ai2i.common.utils.asyncio import custom_gather
from ai2i.common.utils.batch import with_batch
from ai2i.dcollection.data_access_context import DocumentCollectionContext
from ai2i.dcollection.external_api.s2.common import s2_retry
from ai2i.dcollection.interface.collection import Document
from ai2i.dcollection.interface.document import (
    Author,
    Citation,
    CorpusId,
    DocumentFieldName,
    Journal,
    OriginQuery,
    PublicationVenue,
    Snippet,
)

logger = logging.getLogger(__name__)

type S2RestFields = str
DEFAULT_BATCH_SIZE = 20
MAX_TOTAL_CONNECTION_RESULTS = 1000
references_or_citations_fields = [
    "corpusId",
    "referenceCount",
    "citationCount",
    "influentialCitationCount",
    "year",
    "publicationDate",
]


async def from_s2(
    entities: Sequence[Document],
    fields: Sequence[DocumentFieldName],
    context: DocumentCollectionContext,
) -> Sequence[Document]:
    from ai2i.dcollection.collection import keyed_by_corpus_id

    @with_batch(
        batch_size=DEFAULT_BATCH_SIZE,
        max_concurrency=context.s2_max_concurrency,
        force_deterministic=context.force_deterministic,
    )
    @s2_retry()
    async def _batched_from_s2(entities: Sequence[Document], fields: Sequence[DocumentFieldName]) -> Sequence[Document]:
        docs_by_corpus_id = keyed_by_corpus_id(entities)
        try:
            papers = await _fetch_paper_data([int(entity.corpus_id) for entity in entities], fields, context.s2_client)
        except BadQueryParametersException as e:
            if "No valid paper ids given" in str(e):
                logger.warning(
                    f"No valid_paper ids given in {[int(entity.corpus_id) for entity in entities]} (usually means these are just missing from S2)."
                )
                return []
            else:
                raise e
        except Exception as e:
            raise e
        docs, found_corpus_ids = await _to_docs(docs_by_corpus_id, fields, papers)
        await _fill_missing_corpus_ids(
            docs, docs_by_corpus_id, found_corpus_ids, fields, context.s2_client, context.force_deterministic
        )
        return docs

    return list(await _batched_from_s2(entities, fields))


def _document_fields_to_s2_fields(
    fields: Sequence[DocumentFieldName],
) -> Sequence[S2RestFields]:
    s2_fields = [camelize(f, uppercase_first_letter=False) for f in fields]
    if "corpusId" not in s2_fields:
        s2_fields.append("corpusId")
        s2_fields.append("externalIds")
    if "citations" in s2_fields:
        s2_fields.remove("citations")
        s2_fields.append("citationCount")
    if "references" in s2_fields:
        s2_fields.append("referenceCount")
        s2_fields.extend([f"references.{f}" for f in references_or_citations_fields])
    return s2_fields


async def get_paginated_results(
    corpus_id: int,
    references_or_citations: Literal["references", "citations"],
    s2_client: AsyncSemanticScholar,
    overall_limit: int = MAX_TOTAL_CONNECTION_RESULTS,
) -> Paper:
    try:
        if references_or_citations == "references":
            results = await s2_client.get_paper_references(
                f"CorpusId:{corpus_id}", references_or_citations_fields, limit=MAX_TOTAL_CONNECTION_RESULTS
            )
        else:
            results = await s2_client.get_paper_citations(
                f"CorpusId:{corpus_id}", references_or_citations_fields, limit=MAX_TOTAL_CONNECTION_RESULTS
            )
        logger.info(f"Got {len(results)} results on first citations page for {corpus_id}.")
    except TypeError:
        # the TypeError here is due to a null-reference bug in semantic-scholar package.
        # they haven't handled the new case of "SOME field has been elided by the publisher"
        # P.S. this happens much more often for "references" than "citations" for some reason
        logger.info(f"'{references_or_citations}' field has been elided by the publisher for paper {corpus_id}.")
        return Paper({"corpusId": corpus_id, "citations": []})

    if overall_limit > MAX_TOTAL_CONNECTION_RESULTS:
        try:
            while len(results) < overall_limit:
                results.next_page()
        except NoMorePagesException:
            logger.info(f"No more pages for {corpus_id} after fetching {len(results)} results.")
    return Paper({"corpusId": corpus_id, "citations": [item.raw_data for item in results.items]})


async def _fetch_paper_data(
    corpus_ids: Sequence[int],
    fields: Sequence[DocumentFieldName],
    s2_client: AsyncSemanticScholar,
) -> Sequence[Paper]:
    s2_fields = list(_document_fields_to_s2_fields(fields))
    try:
        papers = cast(
            Sequence[Paper],
            await s2_client.get_papers([f"CorpusId:{cid}" for cid in corpus_ids], fields=[*s2_fields]),
        )
    except Exception as e:
        logger.error(f"Failed to fetch paper data for corpus_ids={corpus_ids}: {e}")
        raise e

    if "citations" not in fields:
        return papers

    cid_to_paper = {paper.corpusId: paper for paper in papers}
    corpus_ids_with_few_citations = []
    corpus_ids_with_many_citations = []
    for paper in papers:
        if paper.citationCount <= MAX_TOTAL_CONNECTION_RESULTS:
            corpus_ids_with_few_citations.append(paper.corpusId)
        else:
            corpus_ids_with_many_citations.append(paper.corpusId)

    # first bring all papers with less than 1K citations using get_papers (as its batched!)
    papers_with_few_citations = dict()
    if corpus_ids_with_few_citations:
        try:
            citation_fields = ["corpusId"] + [f"citations.{f}" for f in references_or_citations_fields]
            papers_with_few_citations = {
                paper.corpusId: paper
                for paper in cast(
                    Sequence[Paper],
                    await s2_client.get_papers(
                        [f"CorpusId:{cid}" for cid in corpus_ids_with_few_citations], fields=[*citation_fields]
                    ),
                )
            }
        except Exception as e:
            logger.error(f"Failed to fetch paper citation data (bulk) for corpus_ids={corpus_ids}: {e}")
            raise e

    # then bring all papers with more than 1K citations sequentially using get_paper_citations (as its not batched)
    papers_with_many_citations = dict()
    for cid in corpus_ids_with_many_citations:
        try:
            papers_with_many_citations[cid] = await get_paginated_results(cid, "citations", s2_client)
        except Exception as e:
            logger.warning(
                f"Failed to fetch paper citation data (sequential) for corpus_id={cid}, continue with best effort: {e}"
            )

    # merge
    final_papers = []
    for cid, paper in cid_to_paper.items():
        try:
            paper_with_citation = papers_with_few_citations.get(cid)
            if not paper_with_citation:
                paper_with_citation = papers_with_many_citations.get(cid)
            if not paper_with_citation:
                continue
            final_papers.append(
                Paper({**paper.raw_data, "citations": [p.raw_data for p in paper_with_citation.citations]})
            )
        except Exception:
            logger.warning(f"failed to add citations to paper {cid}. skipping.")
    return final_papers


async def _to_docs(
    docs_by_corpus_id: Mapping[CorpusId, Document],
    fields: Sequence[DocumentFieldName],
    papers: Sequence[Paper],
) -> tuple[list[Document], Sequence[CorpusId]]:
    found_corpus_ids: list[CorpusId] = []
    docs = []
    for paper in papers:
        corpus_id = str(paper.corpusId)
        found_corpus_ids.append(corpus_id)
        if corpus_id not in docs_by_corpus_id:
            logger.warning(
                f"Document fetched from S2 has no corresponding entity: {corpus_id} when trying to load fields={fields}"
            )
            continue
        doc = s2_paper_to_document(corpus_id, paper)
        docs.append(doc)
    return docs, found_corpus_ids


def s2_paper_to_document(
    corpus_id: str,
    paper: Paper | None = None,
    origin_query: OriginQuery | None = None,
    contexts: list[str] | None = None,
) -> Document:
    from ai2i.dcollection import PaperFinderDocument

    if paper:
        doc = PaperFinderDocument(
            corpus_id=str(corpus_id),
            origins=[origin_query] if origin_query else [],
            url=paper.url if hasattr(paper, "_url") else f"https://api.semanticscholar.org/CorpusId:{corpus_id}",
            title=paper.title,
            year=paper.year,
            authors=[Author(name=a.name, author_id=a.authorId) for a in paper.authors or []],
            abstract=paper.abstract,
            venue=paper.venue,
            publication_venue=PublicationVenue.from_dict(paper.publicationVenue) if paper.publicationVenue else None,
            publication_types=paper.publicationTypes,
            tldr=paper.tldr.text if paper.tldr else None,
            citations=[
                Citation(
                    target_corpus_id=int(c.corpusId),
                    citation_count=c.citationCount,
                    reference_count=c.referenceCount,
                    influential_citation_count=c.influentialCitationCount,
                    year=c.year,
                    publication_date=c.publicationDate.date() if c.publicationDate else None,
                )
                for c in paper.citations or []
                if c.corpusId is not None
            ],
            references=[
                Citation(
                    target_corpus_id=int(r.corpusId),
                    citation_count=r.citationCount,
                    reference_count=r.referenceCount,
                    influential_citation_count=r.influentialCitationCount,
                    year=r.year,
                    publication_date=r.publicationDate.date() if r.publicationDate else None,
                )
                for r in paper.references or []
                if r.corpusId is not None
            ],
            citation_count=paper.citationCount,
            reference_count=paper.referenceCount,
            influential_citation_count=paper.influentialCitationCount,
            snippets=[Snippet(text=c) for c in contexts] if contexts else [],
            journal=Journal.from_dict(paper.journal) if paper.journal else None,
            publication_date=paper.publicationDate.date() if paper.publicationDate else None,
        )
    else:
        doc = PaperFinderDocument(corpus_id=str(corpus_id), origins=[origin_query] if origin_query else [])
    return doc


# For the scenario where fetching by corpus_id finds the paper but returns a different corpus_id
# (e.g. could happen for papers with multiple revisions). In that case, we fetch these papers one-by-one,
# so we can map the returned paper to the original corpus_id.
async def _fill_missing_corpus_ids(
    docs: list[Document],
    docs_by_corpus_id: Mapping[CorpusId, Document],
    found_corpus_ids: Sequence[CorpusId],
    fields: Sequence[DocumentFieldName],
    s2_client: AsyncSemanticScholar,
    force_deterministic: bool = False,
) -> None:
    missing_corpus_ids: list[CorpusId] = [
        corpus_id for corpus_id in docs_by_corpus_id.keys() if corpus_id not in found_corpus_ids
    ]
    missing_paper_results_futures = []
    for missing_corpus_id in missing_corpus_ids:
        missing_paper_results_futures.append(
            _s2_get_paper_data_for_missing_id(missing_corpus_id=missing_corpus_id, fields=fields, s2_client=s2_client)
        )
    missing_paper_results = await custom_gather(*missing_paper_results_futures, force_deterministic=force_deterministic)
    for missing_corpus_id, missing_paper_result in missing_paper_results:
        if missing_paper_result:
            missing_doc = s2_paper_to_document(missing_corpus_id, paper=missing_paper_result)
            docs.append(missing_doc)


async def _s2_get_paper_data_for_missing_id(
    missing_corpus_id: CorpusId,
    fields: Sequence[DocumentFieldName],
    s2_client: AsyncSemanticScholar,
) -> tuple[CorpusId, Paper | None]:
    try:
        missing_papers = await _fetch_paper_data([int(missing_corpus_id)], fields, s2_client)
        return missing_corpus_id, missing_papers[0] if missing_papers else None
    except Exception as e:
        logger.error(f"Failed to fetch missing paper {missing_corpus_id} from S2: {e}")
        return missing_corpus_id, None
