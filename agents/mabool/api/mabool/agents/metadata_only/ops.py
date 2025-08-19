import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, Self, Sequence, TypeVar, override

import ai2i.dcollection as dc
from ai2i.config import config_value
from ai2i.dcollection.interface.collection import BASIC_FIELDS, DocumentPredicate
from ai2i.dcollection.interface.document import Author, ExtractedYearlyTimeRange
from ai2i.di import DI
from pydantic import BaseModel, PrivateAttr, model_validator
from semanticscholar.Author import Author as S2Author

from mabool.data_model.config import cfg_schema
from mabool.data_model.rounds import RoundContext
from mabool.utils import context_deps

from . import high

logger = logging.getLogger(__name__)

my_dir = Path(__file__).parent
runs_dir = my_dir / "experiments"
my_mod_name = __name__.split(".")[-1]
vcr_dir = runs_dir / "cassettes" / my_mod_name


OPS_FIELDS = [*BASIC_FIELDS, "publication_venue", "citation_count", "publication_types", "fields_of_study"]


R = TypeVar("R")


class Op(BaseModel, Generic[R]):
    _result: R | None = PrivateAttr(default=None)
    _factory: dc.DocumentCollectionFactory = PrivateAttr()

    def build(self, factory: dc.DocumentCollectionFactory) -> Self:
        """Build the operation with a specific factory."""
        self._factory = factory
        for _, v in self:
            match v:
                case Op():
                    v.build(factory)
                case Iterable():
                    for item in v:
                        if isinstance(item, Op):
                            item.build(factory)
                case _:
                    continue
        return self

    @property
    def factory(self) -> dc.DocumentCollectionFactory:
        return self._factory

    @abstractmethod
    async def run(self) -> R:
        """Run the operation and return a result."""
        pass

    async def __call__(self) -> R:
        if self._result is None:
            logger.debug(f"Started {self}")
            self._result = await self.run()
        return self._result


class DocOp(Op[dc.DocumentCollection], ABC):
    async def __call__(self) -> dc.DocumentCollection:
        result = await super().__call__()
        logger.debug(f"Returned {len(result.documents)} documents.")
        return result


@dataclass
class AuthorsCollection:
    authors: Sequence[Author]


class AuthorOp(Op[AuthorsCollection], ABC):
    async def __call__(self) -> AuthorsCollection:
        result = await super().__call__()
        logger.debug(f"Returned {len(result.authors)} authors.")
        return result


class Plan(DocOp):
    """A stub op"""

    action: str
    depends_on: Sequence[DocOp] = []

    async def run(self) -> dc.DocumentCollection:
        raise NotImplementedError(f"{self.action} is not implemented yet.")


class Union(DocOp):
    items: Sequence[DocOp]

    async def run(self) -> dc.DocumentCollection:
        colls = await asyncio.gather(*(item() for item in self.items))
        return self.factory.merge(colls)


class Intersect(DocOp):
    items: Sequence[DocOp]

    async def run(self) -> dc.DocumentCollection:
        colls = await asyncio.gather(*(item() for item in self.items))
        corpus_ids_sets = [{doc.corpus_id for doc in coll.documents} for coll in colls]
        intersected_corpus_ids = set.intersection(*corpus_ids_sets)
        intersect = self.factory.from_ids(corpus_ids=list(intersected_corpus_ids))
        return await intersect.with_fields(OPS_FIELDS)


class FromS2ByAuthorByName(DocOp):
    author: str

    async def run(self) -> dc.DocumentCollection:
        author_profiles = await self.factory.s2_client().search_author(query=self.author)
        request_context: RoundContext | None = DI.get_dependency(context_deps.request_context)
        by_author = await self.factory.from_s2_by_author(
            authors_profiles=[list(author_profiles)],
            limit=10_000,
            inserted_before=request_context.inserted_before if request_context else None,
        )
        return await by_author.with_fields(OPS_FIELDS)


class FromS2ByAuthorById(DocOp):
    author: Author

    async def run(self) -> dc.DocumentCollection:
        if not self.author.author_id:
            raise ValueError("Author must have an author_id.")
        author_profile = S2Author(data={"authorId": self.author.author_id, "name": self.author.name})
        request_context: RoundContext | None = DI.get_dependency(context_deps.request_context)
        by_author = await self.factory.from_s2_by_author(
            authors_profiles=[[author_profile]],
            limit=10_000,
            inserted_before=request_context.inserted_before if request_context else None,
        )
        return await by_author.with_fields(OPS_FIELDS)


class FromS2ByTitle(DocOp):
    name: str
    time_range: ExtractedYearlyTimeRange | None = None
    venues: list[str] | None = None

    async def run(self) -> dc.DocumentCollection:
        candidates = await self.factory.from_s2_by_title(
            query=self.name,
            time_range=self.time_range,
            venues=self.venues,
        )
        if candidates.documents:
            return candidates
        request_context: RoundContext | None = DI.get_dependency(context_deps.request_context)
        candidates = await self.factory.from_s2_search(
            query=self.name, limit=10, inserted_before=request_context.inserted_before if request_context else None
        )
        results = candidates.filter(lambda doc: (doc.title or "").lower().startswith(self.name.lower() + ":"))
        return await results.with_fields(OPS_FIELDS)


class FromS2Search(DocOp):
    time_range: ExtractedYearlyTimeRange | None = None
    venues: list[str] | None = None
    fields_of_study: list[str] | None = None
    min_citations: int | None = None

    @model_validator(mode="after")
    def _has_any(self) -> Self:
        if not any([self.time_range, self.venues]):
            raise ValueError("At least one of 'time_range' or 'venues' must be provided.")
        return self

    async def run(self) -> dc.DocumentCollection:
        if not self.time_range:
            logger.warning("Searching by venue without a time range may surpass the 10k results limit.")
        request_context: RoundContext | None = DI.get_dependency(context_deps.request_context)
        return await self.factory.from_s2_search(
            query="",
            limit=10_000,
            venues=self.venues,
            time_range=self.time_range,
            fields_of_study=self.fields_of_study,
            min_citations=self.min_citations,
            fields=OPS_FIELDS,
            inserted_before=request_context.inserted_before if request_context else None,
        )


class EnrichWithReferences(DocOp):
    source: DocOp

    async def run(self) -> dc.DocumentCollection:
        source = await self.source()
        return await source.with_fields(["references"])


class FilterCiting(DocOp):
    source: DocOp
    to_cite: DocOp

    async def run(self) -> dc.DocumentCollection:
        docs_that_must_cite = await self.source()
        docs_to_cite = await self.to_cite()
        corpus_ids_to_cite = {doc.corpus_id for doc in docs_to_cite.documents}
        return docs_that_must_cite.filter(
            lambda doc: any(str(ref.target_corpus_id) in corpus_ids_to_cite for ref in doc.references or [])
        )


class FilterCitedBy(DocOp):
    source: DocOp
    that_cite: DocOp

    async def run(self) -> dc.DocumentCollection:
        docs_that_must_be_cited = await self.source()
        docs_that_cite = await self.that_cite()
        corpus_ids_that_are_cited = {
            str(ref.target_corpus_id) for doc in docs_that_cite.documents if doc.references for ref in doc.references
        }
        return docs_that_must_be_cited.filter(lambda doc: doc.corpus_id in corpus_ids_that_are_cited)


class GetAllReferences(DocOp):
    source: DocOp

    async def run(self) -> dc.DocumentCollection:
        source = await self.source()
        source = await source.with_fields(["references"])
        corpus_ids = {str(ref.target_corpus_id) for doc in source.documents if doc.references for ref in doc.references}
        refs = self.factory.from_ids(corpus_ids=list(corpus_ids))
        return await refs.with_fields(OPS_FIELDS)


class GetAllCiting(DocOp):
    """
    Get all documents that cite the documents in the source collection.
    Limited to 10,000 citing documents per source document.
    """

    limit: ClassVar[int] = 10_000
    source: DocOp
    _semaphore: asyncio.Semaphore = PrivateAttr()

    @override
    def build(self, factory: dc.DocumentCollectionFactory) -> Self:
        max_concurrent = config_value(cfg_schema.metadata_planner_agent.ops_max_concurrency)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        return super().build(factory)

    async def task(self, doc: dc.Document) -> dc.DocumentCollection:
        request_context: RoundContext | None = DI.get_dependency(context_deps.request_context)
        async with self._semaphore:
            return await self.factory.from_s2_citing_papers(
                corpus_id=doc.corpus_id,
                total_limit=self.limit,
                inserted_before=request_context.inserted_before if request_context else None,
            )

    async def run(self) -> dc.DocumentCollection:
        source = await self.source()
        # restrict CPS + return excptions from gather
        tasks = [self.task(doc) for doc in source.documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        citing_docs = [doc for doc in results if isinstance(doc, dc.DocumentCollection)]
        return await self.factory.merge(citing_docs).with_fields(OPS_FIELDS)


class FindAuthorByName(AuthorOp):
    author: str

    async def run(self) -> AuthorsCollection:
        author_profiles = await self.factory.s2_client().search_author(query=self.author)
        if not author_profiles:
            raise ValueError(f"No authors found for name: {self.author}")
        authors = [
            Author(
                author_id=profile.authorId,
                name=profile.name,
            )
            for profile in author_profiles
            if profile.authorId and profile.name
        ]
        return AuthorsCollection(authors=authors)


class AuthorsOfPapers(AuthorOp):
    papers: DocOp

    async def run(self) -> AuthorsCollection:
        docs = await self.papers()
        docs_with_authors = await docs.with_fields(["authors"])
        author_by_id = {
            author.author_id: author
            for doc in docs_with_authors.documents
            if doc.authors
            for author in doc.authors
            if author.author_id
        }
        authors = list(author_by_id.values())
        return AuthorsCollection(authors)


class ByAuthorsOfPapers(DocOp):
    all_authors: bool = False
    min_authors_of_papers: int | None = None
    authors: AuthorOp

    @model_validator(mode="after")
    def no_all_authors_and_min_authors(self) -> Self:
        if self.all_authors and self.min_authors_of_papers is not None:
            raise ValueError("Cannot use both 'all_authors' and 'min_authors_of_papers'.")
        return self

    def make_filter(self, author_ids: set[str]) -> DocumentPredicate:
        def filter_min_authors(doc: dc.Document) -> bool:
            assert self.min_authors_of_papers
            if not doc.authors:
                return False
            doc_author_ids = {author.author_id for author in doc.authors if author.author_id}
            authors_of_papers = doc_author_ids.intersection(author_ids)
            return len(authors_of_papers) >= self.min_authors_of_papers

        return filter_min_authors

    async def run(self) -> dc.DocumentCollection:
        authors_collection = await self.authors()
        ops = [FromS2ByAuthorById(author=author) for author in authors_collection.authors]
        op_cls = Intersect if self.all_authors else Union
        op = op_cls(items=ops).build(self.factory)
        res = await op()
        if self.min_authors_of_papers:
            author_ids = {author.author_id for author in authors_collection.authors if author.author_id}
            filter = self.make_filter(author_ids=author_ids)
            return res.filter(filter)
        else:
            return res


class FilterByMinTotalAuthors(DocOp):
    source: DocOp
    min_total_authors: int

    async def run(self) -> dc.DocumentCollection:
        source = await self.source()
        return source.filter(lambda doc: len(doc.authors or []) >= self.min_total_authors)


class FilterByMetadata(DocOp):
    source: DocOp
    years: list[ExtractedYearlyTimeRange] | None = None
    venue: list[str] | None = None
    venue_group: list[str] | None = None
    field_of_study: str | None = None
    publication_types: list[str] | None = None
    min_citations: int | None = None

    def is_in_year_range(self, doc_year: int, year_range: ExtractedYearlyTimeRange) -> bool:
        if year_range.start:
            if doc_year < year_range.start:
                return False
        if year_range.end:
            if doc_year > year_range.end:
                return False
        return True

    def validate_venue(self, doc: dc.Document) -> bool:
        if not self.venue:
            raise ValueError("No venue list to validate")
        venue_names = {venue.lower() for venue in self.venue}
        if doc.venue and doc.venue.lower() in venue_names:
            return True
        if not (doc.publication_venue and doc.publication_venue.alternate_names):
            return False
        for name in doc.publication_venue.alternate_names:
            if name.lower() in venue_names:
                return True
        return False

    def validate_venue_group(self, doc: dc.Document) -> bool:
        if not self.venue_group:
            raise ValueError("No venue group list to validate")
        if not doc.venue:
            return False
        if any(group.lower() in doc.venue.lower() for group in self.venue_group):
            return True
        return False

    def filter(self, doc: dc.Document) -> bool:
        if self.years:
            if not doc.year:
                return False
            if not any(self.is_in_year_range(doc.year, year) for year in self.years):
                return False
        if self.venue:
            if not self.validate_venue(doc):
                return False
        if self.venue_group:
            if not self.validate_venue_group(doc):
                return False
        if self.field_of_study:
            if not doc.fields_of_study:
                return False
            if self.field_of_study.lower() not in (fos.lower() for fos in doc.fields_of_study):
                return False
        if self.publication_types:
            if not doc.publication_types:
                return False
            if not any(
                pub_type.lower() in (pt.lower() for pt in self.publication_types) for pub_type in doc.publication_types
            ):
                return False
        if self.min_citations is not None:
            if doc.citation_count is None or doc.citation_count < self.min_citations:
                return False
        return True

    async def run(self) -> dc.DocumentCollection:
        source = await self.source()
        return source.filter(self.filter)


class FilterExclude(FilterByMetadata):
    """
    Filter documents that match the exclude criteria.
    This is a special case of FilterByMetadata where the criteria are negated.
    """

    author: AuthorOp | None = None
    citing: DocOp | None = None
    cited_by: DocOp | None = None

    def filter(self, doc: dc.Document) -> bool:
        return not super().filter(doc)

    async def run(self) -> dc.DocumentCollection:
        source = await self.source()
        if any(getattr(self, field) is not None for field in FilterByMetadata.model_fields if field != "source"):
            source = source.filter(self.filter)
        if self.author:
            authors = await self.author()
            author_ids = {author.author_id for author in authors.authors if author.author_id}
            source = source.filter(lambda doc: not any(author.author_id in author_ids for author in doc.authors or []))
        if self.citing:
            citing_docs = await self.citing()
            corpus_ids_to_exclude = {doc.corpus_id for doc in citing_docs.documents}
            source = source.filter(
                lambda doc: not any(str(ref.target_corpus_id) in corpus_ids_to_exclude for ref in doc.references or [])
            )
        if self.cited_by:
            cited_by_docs = await self.cited_by()
            corpus_ids_to_exclude = {doc.corpus_id for doc in cited_by_docs.documents}
            source = source.filter(lambda doc: doc.corpus_id not in corpus_ids_to_exclude)
        return source


class FilterByHighlyCited(DocOp):
    source: DocOp

    async def run(self) -> dc.DocumentCollection:
        source = await self.source()
        citation_counts = [doc.citation_count or 0 for doc in source.documents]
        threshold = high.highly_cited_threshold(citation_counts)
        if threshold is None:
            return self.factory.empty()
        return source.filter(lambda doc: doc.citation_count is not None and doc.citation_count >= threshold)
