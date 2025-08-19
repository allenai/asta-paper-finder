import logging
from collections import defaultdict
from typing import Any

import pandas as pd
from ai2i.dcollection import (
    BASIC_FIELDS,
    AssignedField,
    CitationContext,
    DocumentCollection,
    DocumentCollectionSortDef,
    Offset,
    RefMention,
    SentenceOffsets,
    Snippet,
)
from ai2i.di import DI

from mabool.agents.snowball.snowball_utils import add_snowball_origins
from mabool.external_api import external_api_deps
from mabool.external_api.rerank.cohere import RerankScoreDocInput, RerankScoreInput, RerankScorer
from mabool.utils.dc import DC

logger = logging.getLogger(__name__)


def _collect_cited_corpus_ids(doc_collection: DocumentCollection) -> dict[str, Any]:
    cited_corpus_ids: dict[str, Any] = {}
    for doc in doc_collection.documents:
        if not doc.snippets:
            continue
        for sentence in doc.snippets:
            all_cited_corpus_ids = set(
                [ref_mention.matched_paper_corpus_id for ref_mention in (sentence.ref_mentions or [])]
            )
            if all_cited_corpus_ids:
                for cited_corpus_id in sorted(all_cited_corpus_ids):
                    aggs = cited_corpus_ids.get(
                        cited_corpus_id,
                        {
                            "total_sents": 0,
                            "total_hits": 0,
                            "docs": set(),
                            "scores": {},
                        },
                    )
                    aggs["total_sents"] += 1
                    aggs["docs"].add(doc.corpus_id)
                    if sentence.similarity_scores is not None:
                        for score in sentence.similarity_scores:
                            aggs["total_hits"] += 1
                            model_scores = aggs["scores"].get(score.similarity_model_name, [])
                            model_scores.append(score.score)
                            aggs["scores"][score.similarity_model_name] = model_scores

                    cited_corpus_ids[cited_corpus_id] = aggs
    return cited_corpus_ids


def _cited_corpus_ids_to_df(cited_corpus_ids: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    cited_corpus_ids_for_df = []

    clean_model_names = []
    for k, v in cited_corpus_ids.items():
        cited_corpus_id: dict[str, Any] = {}
        cited_corpus_id["cited_corpus_id"] = k
        cited_corpus_id["total_sents"] = v["total_sents"]
        cited_corpus_id["total_hits"] = v["total_hits"]
        cited_corpus_id["total_docs"] = len(v["docs"])
        for model, scores in v["scores"].items():
            clean_model_name = (
                model.replace("DenseDataset(provider='", "")
                .replace("', variant='", "-")
                .replace("', name='", "-")
                .replace("')", "")
            )
            if clean_model_name not in clean_model_names:
                clean_model_names.append(clean_model_name)
            cited_corpus_id[clean_model_name] = max(scores)

        cited_corpus_ids_for_df.append(cited_corpus_id)
    if not cited_corpus_ids_for_df:
        return pd.DataFrame(), []
    df = pd.DataFrame(cited_corpus_ids_for_df)
    df = df.set_index("cited_corpus_id")
    return df, clean_model_names


def _get_cited_corpus_ids_scores_df(
    doc_collection: DocumentCollection,
) -> tuple[pd.DataFrame, list[str]]:
    if not doc_collection.documents:
        return pd.DataFrame(), []
    cited_corpus_ids = _collect_cited_corpus_ids(doc_collection)

    cited_corpus_ids_scores_df, clean_model_names = _cited_corpus_ids_to_df(cited_corpus_ids)
    return cited_corpus_ids_scores_df, clean_model_names


def scoring_func(
    row: pd.Series,
    models: list[str],
    total_hits_bias: float,
    vespa_bias: float,
    abst_bias: float,
    e5_bias: float,
) -> float:
    row = row.fillna(0.0)
    score: float = total_hits_bias * row["total_hits"]
    vespa_models = [model for model in models if "vespa" in model and model in row.index]
    if vespa_models:
        score += vespa_bias * row[vespa_models].median()
    abst_models = [model for model in models if model not in vespa_models and "abst" in model and model in row.index]
    if abst_models:
        score += abst_bias * row[abst_models].median()
    other_models = [  # e5
        model for model in models if model not in vespa_models and model not in abst_models and model in row.index
    ]
    if other_models:
        score += e5_bias * row[other_models].median()

    return score


def get_cited_corpus_ids_scores(
    doc_collection: DocumentCollection,
    total_hits_bias: float = 0.2,
    vespa_bias: float = 1.0,
    abst_bias: float = 0.2,
    e5_bias: float = 0.5,
) -> dict[str, float]:
    df, model_names = _get_cited_corpus_ids_scores_df(doc_collection)
    if df.empty:
        return {}
    df_norm = (df - df.min()) / (df.max() - df.min())
    df_norm["score"] = df_norm.apply(
        lambda row: scoring_func(
            row,
            model_names,
            total_hits_bias=total_hits_bias,
            vespa_bias=vespa_bias,
            abst_bias=abst_bias,
            e5_bias=e5_bias,
        ),
        axis=1,
    )
    df_norm = df_norm.sort_values("score", ascending=False, kind="stable")
    return dict(df_norm["score"])


def split_text_to_sentence_offsets(snippet: Snippet) -> list[SentenceOffsets]:
    sentence_offsets = []
    start = 0
    text = snippet.text.replace(" et. al. ", "<ETX_ALX>").replace(" et. ", "<ETX>").replace(" al. ", "<ALX>")
    snippet_char_start_offset = snippet.char_start_offset or 0
    for i in range(1, len(text) - 1):
        c = text[i]
        prev_c = text[i - 1]
        if c == " " and prev_c in [".", "!", "?"]:
            sentence_offsets.append(
                SentenceOffsets(
                    within_snippet_offset=Offset(start=start, end=i + 1),
                    global_offset=Offset(
                        start=snippet_char_start_offset + start,
                        end=snippet_char_start_offset + i + 1,
                    ),
                )
            )
            start = i + 1

    if start < len(text):
        sentence_offsets.append(
            SentenceOffsets(
                within_snippet_offset=Offset(start=start, end=len(text)),
                global_offset=Offset(
                    start=snippet_char_start_offset + start,
                    end=snippet_char_start_offset + len(text),
                ),
            )
        )

    return sentence_offsets


def _align_ref_mentions(
    snippet: Snippet, sentences: list[Snippet], sentence_offsets: list[SentenceOffsets]
) -> list[Snippet]:
    if not snippet.ref_mentions:
        return sentences

    sentence_idx_to_ref_mentions: dict[int, list[RefMention]] = defaultdict(list)
    for ref in snippet.ref_mentions:
        if not snippet.sentences or not ref.within_snippet_offset_start or not ref.within_snippet_offset_end:
            continue
        for i, offset in enumerate(sentence_offsets):
            if (
                not offset.within_snippet_offset
                or not offset.within_snippet_offset.start
                or not offset.within_snippet_offset.end
            ):
                continue
            if (
                ref.within_snippet_offset_start >= offset.within_snippet_offset.start
                and ref.within_snippet_offset_start <= offset.within_snippet_offset.end
            ) or (
                ref.within_snippet_offset_end >= offset.within_snippet_offset.start
                and ref.within_snippet_offset_end <= offset.within_snippet_offset.end
            ):
                # fix ref offsets
                new_ref: RefMention = RefMention(
                    matched_paper_corpus_id=ref.matched_paper_corpus_id,
                    within_snippet_offset_start=ref.within_snippet_offset_start - offset.within_snippet_offset.start,
                    within_snippet_offset_end=ref.within_snippet_offset_end - offset.within_snippet_offset.start,
                )

                sentence_idx_to_ref_mentions[i].append(new_ref)

    for i, refs in sentence_idx_to_ref_mentions.items():
        sentences[i].ref_mentions = refs

    return sentences


def split_snippets_to_sentence_snippets(snippet: Snippet) -> list[Snippet]:
    if snippet.similarity_scores and any(
        ("bifroest" in score.similarity_model_name) for score in snippet.similarity_scores
    ):
        return [snippet]
    if not snippet.text or snippet.char_start_offset is None:
        return [snippet]
    if not snippet.sentences or not all(o.within_snippet_offset for o in snippet.sentences):
        sentence_offsets = split_text_to_sentence_offsets(snippet)
    else:
        sentence_offsets = snippet.sentences

    sentences: list[Snippet] = []
    for offset in sentence_offsets:
        if (
            not offset.within_snippet_offset
            or offset.within_snippet_offset.start is None
            or offset.within_snippet_offset.end is None
        ):
            continue
        sentence = Snippet(
            text=snippet.text[offset.within_snippet_offset.start : offset.within_snippet_offset.end],
            similarity_scores=snippet.similarity_scores,
            char_start_offset=(snippet.char_start_offset + offset.within_snippet_offset.start),
            char_end_offset=(snippet.char_start_offset + offset.within_snippet_offset.end),
        )
        sentences.append(sentence)

    if not snippet.ref_mentions:
        return sentences

    # align ref_mentions
    sentences = _align_ref_mentions(snippet, sentences, sentence_offsets)

    return sentences


def _flatten(lst: list[Any]) -> list[Any]:
    return [item for sublist in lst for item in sublist]


def get_citation_contexts_from_snippets(doc_collection: DocumentCollection) -> dict[str, list[CitationContext]]:
    sents_with_ref_mentions = defaultdict(list)
    for doc in doc_collection.documents:
        if not doc.snippets:
            continue
        sentences = _flatten([split_snippets_to_sentence_snippets(snippet) for snippet in doc.snippets])
        for sentence in sentences:
            if sentence.ref_mentions:
                for ref in sentence.ref_mentions:
                    sents_with_ref_mentions[ref.matched_paper_corpus_id].append(
                        CitationContext(
                            text=sentence.text,
                            source_corpus_id=doc.corpus_id,
                            within_snippet_offset_start=ref.within_snippet_offset_start,
                            within_snippet_offset_end=ref.within_snippet_offset_end,
                            similarity_score=max([score.score for score in sentence.similarity_scores])
                            if sentence.similarity_scores
                            else 0.0,
                        )  # NOTE: this will behave weirdly with multiple model scores
                    )
    # sort each list of sentences by score
    return {
        k: [s for s in sorted(v, key=lambda x: x.similarity_score, reverse=True)]
        for k, v in sents_with_ref_mentions.items()
    }


@DI.managed
async def rerank_citation_contexts(
    papers: dict[str, list[CitationContext]],
    input_query: str,
    reranker: RerankScorer = DI.requires(external_api_deps.rerank_scorer),
) -> dict[str, float]:
    try:
        rerank_input = RerankScoreInput(
            query=input_query,
            docs=[
                RerankScoreDocInput(
                    corpus_id=corpus_id,
                    text="\n".join([c.text for c in contexts if c]),
                )
                for corpus_id, contexts in papers.items()
                if contexts
            ],
        )
        rerank_results = await reranker.rerank(rerank_input=rerank_input)
        return {r.corpus_id: r.score for r in rerank_results.results}
    except Exception as e:
        logger.exception(f"Failed to rerank using cohere: {e}")
        raise e


async def get_top_cited_corpus_ids(
    cited_corpus_ids_scores: dict[str, float],
    citation_contexts: dict[str, list[CitationContext]],
    query: str,
    top_k: int,
    fast_mode: bool,
    rerank_score_threshold: float = 0.5,
) -> tuple[list[str], dict[str, float]]:
    if not fast_mode:
        top_corpus_ids_for_rerank = sorted(
            cited_corpus_ids_scores, key=lambda k: cited_corpus_ids_scores[k], reverse=True
        )[:1000]
        try:
            rerank_scores = await rerank_citation_contexts(
                {k: citation_contexts[k] for k in top_corpus_ids_for_rerank},
                input_query=query,
            )
        except Exception as e:
            logger.exception(f"Failed to rerank citation contexts: {e}")
            rerank_scores = {}

        if rerank_scores and not all(score == 0.0 for score in rerank_scores.values()):
            cited_corpus_ids_scores = rerank_scores

        cited_corpus_ids_scores = {k: v for k, v in cited_corpus_ids_scores.items() if v > rerank_score_threshold}

    top_corpus_ids = sorted(cited_corpus_ids_scores, key=lambda k: (cited_corpus_ids_scores[k], k), reverse=True)[
        :top_k
    ]
    return top_corpus_ids, cited_corpus_ids_scores


async def run_snippet_snowball(
    content_query: str,
    doc_collection: DocumentCollection,
    top_k: int,
    search_iteration: int,
    fast_mode: bool = False,
) -> DocumentCollection:
    try:
        logger.info("=== Snippet snowball ===")
        citation_contexts = get_citation_contexts_from_snippets(doc_collection)
        cited_corpus_ids_scores = get_cited_corpus_ids_scores(doc_collection)
        cited_corpus_ids_scores = {
            k: v for k, v in cited_corpus_ids_scores.items() if k in citation_contexts and citation_contexts[k]
        }
        if not cited_corpus_ids_scores or not citation_contexts:
            return DC.empty()

        top_cited_corpus_ids, cited_corpus_ids_scores = await get_top_cited_corpus_ids(
            cited_corpus_ids_scores, citation_contexts, content_query, top_k, fast_mode
        )

        top_cited_docs = DC.from_ids(top_cited_corpus_ids)
        top_cited_docs = await top_cited_docs.with_fields(BASIC_FIELDS)

        # add citation contexts
        top_cited_docs = top_cited_docs.map(
            lambda doc: doc.clone_with({"citation_contexts": citation_contexts.get(doc.corpus_id, [])})
        )

        # add scores
        top_cited_docs = await top_cited_docs.with_fields(
            [
                AssignedField[float](
                    field_name="snippet_snowball_score",
                    assigned_values=[
                        cited_corpus_ids_scores.get(doc.corpus_id, 0.0) for doc in top_cited_docs.documents
                    ],
                ),
            ]
        )
        top_cited_docs = top_cited_docs.sorted(
            [DocumentCollectionSortDef(field_name="snippet_snowball_score", order="desc")]
        )
        top_cited_docs = add_snowball_origins(top_cited_docs, variant="snippets", search_iteration=search_iteration)
        return top_cited_docs
    except Exception as e:
        logger.exception(f"Failed to run snippet snowball: {e}")
        return DC.empty()
