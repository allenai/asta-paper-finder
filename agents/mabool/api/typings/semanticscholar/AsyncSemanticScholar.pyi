from typing import Literal

from _typeshed import Incomplete
from semanticscholar.ApiRequester import ApiRequester as ApiRequester
from semanticscholar.Author import Author as Author
from semanticscholar.BaseReference import BaseReference as BaseReference
from semanticscholar.Citation import Citation as Citation
from semanticscholar.PaginatedResults import PaginatedResults as PaginatedResults
from semanticscholar.Paper import Paper as Paper
from semanticscholar.Reference import Reference as Reference

logger: Incomplete

class AsyncSemanticScholar:
    DEFAULT_API_URL: str
    BASE_PATH_GRAPH: str
    BASE_PATH_RECOMMENDATIONS: str
    auth_header: Incomplete
    api_url: Incomplete
    def __init__(
        self, timeout: int = 30, api_key: str = None, api_url: str = None, debug: bool = False, retry: bool = True
    ) -> None: ...
    @property
    def timeout(self) -> int: ...
    @timeout.setter
    def timeout(self, timeout: int) -> None: ...
    @property
    def debug(self) -> bool: ...
    @debug.setter
    def debug(self, debug: bool) -> None: ...
    @property
    def retry(self) -> bool: ...
    @retry.setter
    def retry(self, retry: bool) -> None: ...
    async def get_paper(self, paper_id: str, fields: list = None) -> Paper: ...
    async def get_papers(
        self, paper_ids: list[str], fields: list = None, return_not_found: bool = False
    ) -> list[Paper] | tuple[list[Paper], list[str]]: ...
    async def get_paper_authors(self, paper_id: str, fields: list = None, limit: int = 100) -> PaginatedResults: ...
    async def get_paper_citations(self, paper_id: str, fields: list = None, limit: int = 100) -> PaginatedResults: ...
    async def get_paper_references(self, paper_id: str, fields: list = None, limit: int = 100) -> PaginatedResults: ...
    async def search_paper(
        self,
        query: str,
        year: str = None,
        publication_types: list = None,
        open_access_pdf: bool = None,
        venue: list = None,
        fields_of_study: list = None,
        fields: list = None,
        publication_date_or_year: str = None,
        min_citation_count: int = None,
        limit: int = 100,
        bulk: bool = False,
        sort: str = None,
        match_title: bool = False,
    ) -> PaginatedResults | Paper: ...
    async def get_author(self, author_id: str, fields: list = None) -> Author: ...
    async def get_authors(
        self, author_ids: list[str], fields: list = None, return_not_found: bool = False
    ) -> list[Author] | tuple[list[Author], list[str]]: ...
    async def get_author_papers(self, author_id: str, fields: list = None, limit: int = 100) -> PaginatedResults: ...
    async def search_author(
        self, query: str, fields: list = None, limit: int = 100, max_results: int = 1000
    ) -> PaginatedResults: ...
    async def get_recommended_papers(
        self, paper_id: str, fields: list = None, limit: int = 100, pool_from: Literal["recent", "all-cs"] = "recent"
    ) -> list[Paper]: ...
    async def get_recommended_papers_from_lists(
        self, positive_paper_ids: list[str], negative_paper_ids: list[str] = None, fields: list = None, limit: int = 100
    ) -> list[Paper]: ...
