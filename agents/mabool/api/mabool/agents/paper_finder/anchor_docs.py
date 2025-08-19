from ai2i.dcollection import CorpusId, DenseDataset, DocumentCollection, Snippet

from mabool.agents.paper_finder.definitions import PaperFinderInput
from mabool.data_model.agent import AnalyzedQuery, PartiallyAnalyzedQuery
from mabool.utils.asyncio import custom_gather
from mabool.utils.dc import DC


async def enrich_anchor_documents(
    analyzed_input: AnalyzedQuery | PartiallyAnalyzedQuery,
    inputs: PaperFinderInput,
) -> DocumentCollection:
    if not list(filter(None, inputs.anchor_corpus_ids)):
        return DC.empty()
    anchor_docs_with_similar_snippets = DC.merge(
        await custom_gather(
            *[
                _enrich_anchor_document(query=analyzed_input.content or inputs.query, corpus_id=corpus_id)
                for corpus_id in inputs.anchor_corpus_ids
            ]
        )
    )
    anchor_with_markdown = await anchor_docs_with_similar_snippets.with_fields(["markdown"])
    return anchor_with_markdown


async def _enrich_anchor_document(query: str, corpus_id: CorpusId) -> DocumentCollection:
    return await DC.from_dense_retrieval(
        queries=[query],
        search_iteration=0,
        top_k=2,
        dataset=DenseDataset(provider="vespa", name="open-nora", variant="pa1-v1"),
        corpus_ids=[corpus_id],
    )


def _filter_similar_snippets(snippets: list[Snippet] | None, similarity_threshold: float) -> list[Snippet] | None:
    return (
        [s for s in snippets if max([float(ss) for ss in s.similarity_scores or []]) >= similarity_threshold]
        if snippets
        else None
    )
