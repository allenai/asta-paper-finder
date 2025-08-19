from __future__ import annotations

import dataclasses
import logging
import math
from datetime import datetime
from typing import Any, Sequence

import pandas as pd
from ai2i.config import ufv
from ai2i.dcollection import (
    AggTransformComputedField,
    AssignedField,
    BaseComputedField,
    ComputedField,
    Document,
    DocumentCollection,
    DocumentCollectionSortDef,
    DocumentFieldName,
    PaperFinderDocument,
    RelevanceJudgement,
    Typed,
)

from mabool.agents.common.computed_fields.fields import rerank_score_field
from mabool.agents.common.computed_fields.relevance import relevance_judgement_field
from mabool.data_model.agent import AnalyzedQuery, RelevanceCriteria
from mabool.data_model.ufs import uf
from mabool.utils.dc import DC
from mabool.utils.text import AND_CONNECTOR

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SortPreferences:
    is_recent_explicit: bool = False
    recent_first: bool = False
    recent_last: bool = False
    is_central_explicit: bool = False
    central_first: bool = False
    central_last: bool = False
    consider_content_relevance: bool = True
    relevance_criteria: RelevanceCriteria | None = None

    @staticmethod
    def from_analyzed_query(
        analyzed_query: AnalyzedQuery,
        *,
        assume_recent_first: bool,
        assume_central_first: bool,
        consider_content_relevance: bool,
    ) -> SortPreferences:
        return SortPreferences(
            is_recent_explicit=(
                analyzed_query.extracted_properties.recent_first or analyzed_query.extracted_properties.recent_last
            ),
            recent_first=analyzed_query.extracted_properties.recent_first or assume_recent_first,
            recent_last=analyzed_query.extracted_properties.recent_last,
            is_central_explicit=(
                analyzed_query.extracted_properties.central_first or analyzed_query.extracted_properties.central_last
            ),
            central_first=analyzed_query.extracted_properties.central_first or assume_central_first,
            central_last=analyzed_query.extracted_properties.central_last,
            consider_content_relevance=consider_content_relevance,
            relevance_criteria=analyzed_query.relevance_criteria,
        )

    def required_fields(self) -> list[DocumentFieldName | BaseComputedField[DocumentFieldName, Any]]:
        if (
            self.consider_content_relevance
            and self.relevance_criteria
            and self.relevance_criteria.required_relevance_critieria
        ):
            fields: list[DocumentFieldName | BaseComputedField[DocumentFieldName, Any]] = [
                relevance_judgement_field(self.relevance_criteria),
                rerank_score_field(self.relevance_criteria),
            ]
        else:
            fields = []
        if self.recent_first or self.recent_last:
            fields.append("year")
        if self.central_first or self.central_last:
            fields.append("citation_count")
        return fields

    def get_scoring_weights(self) -> ScoreWeights:
        if self.consider_content_relevance:
            if self.is_recent_explicit and self.is_central_explicit:
                return ScoreWeights(Wcontent_relevance=0.8, Wrecent=0.1, Wcentral=0.1)
            elif self.is_recent_explicit:
                return ScoreWeights(Wcontent_relevance=0.8, Wrecent=0.175, Wcentral=0.025)
            elif self.is_central_explicit:
                return ScoreWeights(Wcontent_relevance=0.8, Wrecent=0.025, Wcentral=0.175)
            else:
                return ScoreWeights(Wcontent_relevance=0.95, Wrecent=0.025, Wcentral=0.025)
        else:
            if self.is_recent_explicit and self.is_central_explicit:
                return ScoreWeights(Wcontent_relevance=0.0, Wrecent=0.5, Wcentral=0.5)
            elif self.is_recent_explicit:
                return ScoreWeights(Wcontent_relevance=0.0, Wrecent=0.9, Wcentral=0.1)
            elif self.is_central_explicit:
                return ScoreWeights(Wcontent_relevance=0.0, Wrecent=0.1, Wcentral=0.9)
            else:
                return ScoreWeights(Wcontent_relevance=0.0, Wrecent=0.5, Wcentral=0.5)

    def get_sorting_weights(self) -> Weights:
        return get_sorting_weights(
            is_recent_explicit=self.is_recent_explicit,
            recent_first=self.recent_first,
            recent_last=self.recent_last,
            is_central_explicit=self.is_central_explicit,
            central_first=self.central_first,
            central_last=self.central_last,
        )

    def get_sorting_explanation(self) -> str:
        return get_sorting_explanation(
            is_recent_explicit=self.is_recent_explicit,
            recent_first=self.recent_first,
            recent_last=self.recent_last,
            is_central_explicit=self.is_central_explicit,
            central_first=self.central_first,
            central_last=self.central_last,
        )


@dataclasses.dataclass
class Weights:
    year: float
    citation_count: float
    rerank_score: float
    num_snippets: float
    original_order: float


def weight_on_similar() -> Weights:
    return Weights(year=0.1, citation_count=0.1, rerank_score=1, num_snippets=0.5, original_order=1)


def weight_on_recent() -> Weights:
    return Weights(year=1, citation_count=0.1, rerank_score=0.1, num_snippets=0.05, original_order=0.1)


def weight_on_central() -> Weights:
    return Weights(year=0.1, citation_count=1, rerank_score=0.1, num_snippets=0.05, original_order=0.1)


def weight_on_recent_and_central_weights() -> Weights:
    return Weights(year=1, citation_count=1, rerank_score=0.1, num_snippets=0.05, original_order=0.1)


def normalize_column(s: pd.Series[float], s_min: float | None = None, s_max: float | None = None) -> pd.Series | float:
    if s_min is None:
        s_min = s.min()
    if s_max is None:
        s_max = s.max()
    if s_max == s_min:
        return 1.0
    return (s - s_min) / (s_max - s_min)


def weighted_average_sort(
    df: pd.DataFrame,
    weights_config: Weights | dict[str, float],
    drop_intermediate_columns: bool = True,
    keep_final_sort_score: bool = True,
    rank_scores: bool = False,
    normalize_ignore_outliers: bool = True,
) -> pd.DataFrame:
    """
    Sort the dataframe by multiple keys with weights.
    The sorting works using a weighted average over the normalized_scores
    rank_scores: if True, use rank scores instead of the actual scores, useful if score values are weirdly distributed
    """
    if isinstance(weights_config, Weights):
        keys, weights = zip(*(dataclasses.asdict(weights_config).items()))
    else:
        keys, weights = zip(*(weights_config.items()))
    for key in keys:
        if rank_scores:
            df[f"__{key}_score"] = df[key].rank(method="dense", na_option="bottom")
        else:
            df[f"__{key}_score"] = df[key]

    # multiply by the weight's sign, from now on, higher is better
    for key, weight in zip(keys, weights):
        df[f"__{key}_score"] = df[f"__{key}_score"] * math.copysign(1, weight)

    # fill nan values with "worst" value
    for key in keys:
        # if all values are NaN, fill with 0
        if df[f"__{key}_score"].isnull().all():
            logger.warning(f"All values for {key} are NaN, filling with 0 for sorting")
            worst_value = df[f"__{key}_score"] = 0
        else:
            worst_value = df[f"__{key}_score"].min()
        df[f"__{key}_score"] = df[f"__{key}_score"].fillna(worst_value)

    # normalize the scores
    for key in keys:
        if normalize_ignore_outliers:
            # key_min = df[f"__{key}_score"].quantile(0.01) # ignore the bottom 1% of the scores
            key_min = df[f"__{key}_score"].min()  # only ignoring the top scores
            key_max = df[f"__{key}_score"].quantile(0.99)
            # clip the scores to ignore outliers
            df[f"__{key}_score"] = df[f"__{key}_score"].apply(lambda x: min(max(x, key_min), key_max))
        else:
            key_min = df[f"__{key}_score"].min()
            key_max = df[f"__{key}_score"].max()
        df[f"__{key}_score"] = normalize_column(df[f"__{key}_score"], key_min, key_max)
    # multiply by weight's absolut value (as we already used the sign)
    # this is done here because needs to happen after normalization to take effect
    for key, weight in zip(keys, weights):
        df[f"__{key}_score"] = df[f"__{key}_score"] * abs(weight)

    df["sorting_score"] = df[[f"__{key}_score" for key in keys]].sum(axis=1)
    df["sorting_rank"] = df["sorting_score"].rank(method="dense", ascending=False)
    df = df.sort_values("sorting_rank", kind="stable")

    if keep_final_sort_score:
        df["final_sort_score"] = df["sorting_score"]
        df["final_sort_score"] = normalize_column(df["final_sort_score"])

        # fill final_sort_score with 0 if it's NaN
        df["final_sort_score"] = df["final_sort_score"].fillna(0.0)

    if drop_intermediate_columns:
        df.drop(columns=[f"__{key}_score" for key in keys], inplace=True)
        df.drop(columns=["sorting_score"], inplace=True)
        df.drop(columns=["sorting_rank"], inplace=True)

    return df


def get_sorting_explanation(
    is_recent_explicit: bool,
    recent_first: bool,
    recent_last: bool,
    is_central_explicit: bool,
    central_first: bool,
    central_last: bool,
) -> str:
    sorting_details = []
    if (recent_first or recent_last) and is_recent_explicit:
        if recent_first:
            sorting_details.append(ufv(uf.sorting.explanation.most_recent))
        elif recent_last:
            sorting_details.append(ufv(uf.sorting.explanation.earliest))
    if (central_first or central_last) and is_central_explicit:
        if central_first:
            sorting_details.append(ufv(uf.sorting.explanation.highly_cited))
        elif central_last:
            sorting_details.append(ufv(uf.sorting.explanation.least_cited))
    sorting_explanation = AND_CONNECTOR.join(sorting_details)
    if sorting_explanation:
        sorting_explanation = ufv(uf.sorting.explanation.sorting_explanation, sorting_explanation=sorting_explanation)
    return sorting_explanation


def get_sorting_weights(
    is_recent_explicit: bool = False,
    recent_first: bool = False,
    recent_last: bool = False,
    is_central_explicit: bool = False,
    central_first: bool = False,
    central_last: bool = False,
) -> Weights:
    if is_recent_explicit and is_central_explicit:
        base_weights = weight_on_recent_and_central_weights()
    elif is_recent_explicit:
        base_weights = weight_on_recent()
    elif is_central_explicit:
        base_weights = weight_on_central()
    else:
        base_weights = weight_on_similar()

    if recent_first:
        recency_weights = dataclasses.replace(base_weights, year=abs(base_weights.year))
    elif recent_last:
        recency_weights = dataclasses.replace(base_weights, year=-abs(base_weights.year))
    else:
        recency_weights = dataclasses.replace(base_weights, year=0)
    if central_first:
        weights = dataclasses.replace(recency_weights, citation_count=abs(recency_weights.citation_count))
    elif central_last:
        weights = dataclasses.replace(recency_weights, citation_count=-abs(recency_weights.citation_count))
    else:
        weights = dataclasses.replace(recency_weights, citation_count=0)

    weights = dataclasses.replace(weights, original_order=0)

    return weights


async def sorted_docs_by_preferences(
    docs: DocumentCollection,
    sort_documents_input: SortPreferences,
    only_use_existing_fields: bool = False,
) -> DocumentCollection:
    docs = await add_broad_search_score(docs, sort_documents_input, only_use_existing_fields=only_use_existing_fields)

    docs = docs.sorted([DocumentCollectionSortDef(field_name="broad_search_score", order="desc")])
    return docs


async def weighted_sort_calculation(
    documents: Sequence[Document],
    weights: Weights | dict[str, float],
) -> Sequence[float]:
    logger.info("Calculating weighted sort score")
    weights_keys = list(weights.keys() if isinstance(weights, dict) else dataclasses.asdict(weights).keys())

    if not documents:
        return []

    df = DC.from_docs(documents).to_dataframe(["corpus_id", *weights_keys])
    df = weighted_average_sort(df, weights)
    weighted_sort_scores: list[float] = []
    for doc in documents:
        doc_rows = df.loc[df["corpus_id"] == doc.corpus_id]
        if doc_rows.empty:
            logger.warning(f"Document {doc.corpus_id} not found in the weighted sort dataframe, filling with 0")
            weighted_sort_score = 0.0
        elif len(doc_rows) > 1:
            logger.warning(
                f"Document {doc.corpus_id} found multiple times in the weighted sort dataframe, using the first one"
            )
            weighted_sort_score = doc_rows["final_sort_score"].iloc[0]
        else:
            weighted_sort_score = doc_rows["final_sort_score"].iloc[0]
        weighted_sort_score = (
            float(weighted_sort_score)
            if isinstance(weighted_sort_score, float) or isinstance(weighted_sort_score, int)
            else 0.0
        )
        weighted_sort_scores.append(weighted_sort_score)
    return weighted_sort_scores


async def add_weighted_sort_score(
    collection: DocumentCollection,
    sort_documents_input: SortPreferences,
) -> DocumentCollection:
    weights = sort_documents_input.get_sorting_weights()
    logger.info(f"Sorting weights: {weights}")
    collection_with_num_snippets = await collection.with_fields(
        [
            ComputedField(
                field_name="num_snippets",
                required_fields=["snippets"],
                computation_func=Typed[PaperFinderDocument, int](lambda doc: len(doc.snippets) if doc.snippets else 0),
            ),
        ],
    )

    docs = collection_with_num_snippets.documents
    original_sort_d = {d.corpus_id: len(docs) - i for i, d in enumerate(docs)}

    collection_with_original_order = await collection_with_num_snippets.with_fields(
        [
            rerank_score_field(sort_documents_input.relevance_criteria),
            AssignedField[int](
                field_name="original_order",
                assigned_values=[original_sort_d[doc.corpus_id] for doc in docs],
            ),
        ],
    )

    async def weighted_sort_calculation_partial(documents: Sequence[Document]) -> Sequence[float]:
        return await weighted_sort_calculation(documents, weights)

    collection_with_final_sort_field = await collection_with_original_order.with_fields(
        [
            AggTransformComputedField[float](
                field_name="weighted_sort_score",
                computation_func=weighted_sort_calculation_partial,
                required_fields=[
                    "corpus_id",
                    "rerank_score",
                    "citation_count",
                    "year",
                    "num_snippets",
                    "original_order",
                ],
            ),
        ],
    )

    return collection_with_final_sort_field


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def sigmoid(x: float, center_shift: float = 0.0, steepness: float = 1.0) -> float:
    return _sigmoid((x + center_shift) * steepness)


def central_first_score_sigmoid(
    citation_count: int | None,
    center_citation_count: int = 50,
    steepness: float = 1.8,
) -> float:
    if not citation_count or citation_count <= 0 or center_citation_count <= 0:
        return 0.0
    try:
        return sigmoid(
            math.log(citation_count + 1),
            center_shift=-math.log(center_citation_count + 1),
            steepness=steepness,
        )
    except Exception as e:
        logger.exception(f"Error in central_first_score_sigmoid: citation_count={citation_count}, {e}")
        return 0.0


def central_last_score_sigmoid(
    citation_count: int | None,
    center_citation_count: int = 50,
    steepness: float = 1.8,
) -> float:
    if not citation_count or citation_count <= 0 or center_citation_count <= 0:
        return 1.0

    return 1 - central_first_score_sigmoid(
        citation_count,
        center_citation_count=center_citation_count,
        steepness=steepness,
    )


def recent_first_score_sigmoid(
    year: int | None, year_max: int | None = None, center_shift: float = 7.0, steepness: float = 0.7
) -> float:
    if year_max is None:
        year_max = datetime.now().year
    if not year or year <= 0:
        return 0.0
    if year >= year_max:
        return 1.0
    try:
        return sigmoid(year, -(year_max - center_shift), steepness)
    except Exception as e:
        logger.exception(f"Error in recent_first_score_sigmoid: year={year}, year_max={year_max}, {e}")
        return 0.0


def recent_last_score_sigmoid(
    year: int | None,
    year_max: int | None = None,
    center_shift: float = 20.0,
    steepness: float = 0.15,
) -> float:
    if year_max is None:
        year_max = datetime.now().year
    if year is None or year >= year_max:
        return 0.0
    if year <= 0:
        return 1.0
    try:
        return 1 - sigmoid(year, -(year_max - center_shift), steepness)
    except Exception as e:
        logger.exception(f"Error in recent_last_score_sigmoid: year={year}, year_max={year_max}, {e}")
        return 0.0


def num_snippets_score_sigmoid(num_snippets: int) -> float:
    if num_snippets <= 0:
        return 0.0
    return sigmoid(math.log(num_snippets**2 + 0.1), center_shift=0.0, steepness=1.0)


@dataclasses.dataclass
class ScoreWeights:
    Wcontent_relevance: float
    Wrecent: float
    Wcentral: float


def content_relevance_score(
    relevance_judgement_score: float, rerank_score: float, num_snippets: int, original_order_score: float = 0.0
) -> float:
    relevance_judgement_score = max(relevance_judgement_score, 0.0)
    rerank_score = max(rerank_score, 0.0)
    num_snippets = max(num_snippets, 0)

    rj_weight = 0.9
    rerank_weight = 0.075
    dense_weight = 0.025

    if original_order_score:
        original_order_weight = dense_weight
        rj_weight -= original_order_weight
    else:
        original_order_weight = 0
        original_order_score = 0

    return (
        rj_weight * relevance_judgement_score
        + rerank_weight * rerank_score
        + dense_weight * num_snippets_score_sigmoid(num_snippets)
        + original_order_weight * original_order_score
    )


def weighted_sum(scores: list[float], weights: list[float]) -> float:
    return sum([score * weight for score, weight in zip(scores, weights)])


def _relevance_judgement_score(relevance_judgement: RelevanceJudgement | None) -> float:
    if relevance_judgement is None:
        return 0.0
    if relevance_judgement.relevance_score is not None:
        return relevance_judgement.relevance_score
    return max(relevance_judgement.relevance / 3, 0.0)


async def add_broad_search_score(
    collection: DocumentCollection,
    sort_documents_input: SortPreferences,
    only_use_existing_fields: bool = False,
) -> DocumentCollection:
    if not only_use_existing_fields:
        required_fields = sort_documents_input.required_fields()
        try:
            collection = await collection.with_fields(required_fields)
        except Exception as e:
            logger.exception(f"Failed to add fields to the documents for sorting, sorting with exisiting fields: {e}")

    recency_score_func = recent_last_score_sigmoid if sort_documents_input.recent_last else recent_first_score_sigmoid
    centrality_score_func = (
        central_last_score_sigmoid if sort_documents_input.central_last else central_first_score_sigmoid
    )
    original_order_scores = {}

    collection = await collection.with_fields(
        [
            AssignedField[float](
                field_name="content_relevance_score",
                required_fields=[],
                assigned_values=[
                    content_relevance_score(
                        relevance_judgement_score=_relevance_judgement_score(doc.relevance_judgement),
                        rerank_score=doc.rerank_score if doc.rerank_score else 0.0,
                        num_snippets=(len(doc.snippets) if doc.snippets else 0)
                        + (len(doc.citation_contexts) if doc.citation_contexts else 0),
                        original_order_score=original_order_scores.get(doc.corpus_id) or 0.0,
                    )
                    for doc in collection.documents
                ],
            ),
            AssignedField[float](
                field_name="recency_score",
                required_fields=[],
                assigned_values=[recency_score_func(doc.year) for doc in collection.documents],
            ),
            AssignedField[float](
                field_name="centrality_score",
                required_fields=[],
                assigned_values=[centrality_score_func(doc.citation_count) for doc in collection.documents],
            ),
        ]
    )

    weights = sort_documents_input.get_scoring_weights()
    collection = await collection.with_fields(
        [
            AssignedField[float](
                field_name="broad_search_score",
                required_fields=[],
                assigned_values=[
                    weighted_sum(
                        [doc["content_relevance_score"], doc["recency_score"], doc["centrality_score"]],
                        [weights.Wcontent_relevance, weights.Wrecent, weights.Wcentral],
                    )
                    for doc in collection.documents
                ],
            )
        ]
    )
    return collection
