from semanticscholar.Author import Author

from ai2i.dcollection.data_access_context import DocumentCollectionContext
from ai2i.dcollection.external_api.s2.common import AsyncPaginatedResults, s2_retry

AUTHORS_PER_CALL_LIMIT = 100
AUTHORS_TOTAL_LIMIT = 1000
AUTHOR_PAPERS_PER_CALL_LIMIT = 1000


@s2_retry()
async def s2_get_authors_by_name(query: str, context: DocumentCollectionContext) -> list[Author]:
    fields = ["authorId", "name", "paperCount", "hIndex"]
    search_author_result = await context.s2_client.search_author(
        query=query.replace("-", ""),
        fields=fields,
        limit=AUTHORS_PER_CALL_LIMIT,
        max_results=AUTHORS_TOTAL_LIMIT,
    )
    return [
        author
        async for author in AsyncPaginatedResults[Author](results=search_author_result, max_results=AUTHORS_TOTAL_LIMIT)
    ]
