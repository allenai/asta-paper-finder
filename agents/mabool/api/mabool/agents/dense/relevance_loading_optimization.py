from __future__ import annotations

import dataclasses
import logging
import random
import textwrap
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from ai2i.config import ConfigValue, config_value, configurable
from ai2i.dcollection import (
    AdaptiveLoader,
    CorpusId,
    Document,
    DocumentCollection,
    DocumentCollectionFactory,
    DocumentFieldName,
    to_reward,
)
from ai2i.di import DI
from mabwiser.mab import LearningPolicy, LearningPolicyType
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from mabool.data_model.config import cfg_schema
from mabool.utils import dc_deps
from mabool.utils.dc import DC

logger = logging.getLogger(__name__)
random.seed(42)


class Shortcircuit(ABC):
    @abstractmethod
    def accumulate(self, docs: list[Document]) -> Shortcircuit:
        pass

    @abstractmethod
    def should_break(self) -> bool:
        pass


class NoopShortcircuit(Shortcircuit):
    def accumulate(self, docs: list[Document]) -> Shortcircuit:
        return self

    def should_break(self) -> bool:
        return False


@dataclasses.dataclass
class HighlyRelevantShortcircuit(Shortcircuit):
    score_cap: int
    doc_ids: set[CorpusId] = dataclasses.field(default_factory=set)
    accumulated_score: float = 0
    found_perfectly_relevant: bool = False

    def accumulate(self, docs: list[Document]) -> Shortcircuit:
        for doc in docs:
            if (
                doc.relevance_judgement is not None
                and doc.relevance_judgement.relevance >= 2
                and doc.corpus_id not in self.doc_ids
            ):
                self.doc_ids.add(doc.corpus_id)
                self.accumulated_score += self._relevance_score_for_cap(doc)
            if doc.relevance_judgement is not None and doc.relevance_judgement.relevance == 3:
                self.found_perfectly_relevant = True
        return self

    def should_break(self) -> bool:
        return self.found_perfectly_relevant and self.accumulated_score >= self.score_cap

    @classmethod
    def _relevance_score_for_cap(cls, doc: Document) -> float:
        if doc.relevance_judgement is not None and doc.relevance_judgement.relevance == 3:
            return 2
        elif doc.relevance_judgement is not None and doc.relevance_judgement.relevance == 2:
            return 1
        else:
            return 0


@DI.managed
async def adaptive_load(
    documents: DocumentCollection,
    field_name: DocumentFieldName = "relevance_judgement",
    docs_quota: int = ConfigValue(cfg_schema.relevance_judgement.quota),
    shortcircuits: Sequence[Shortcircuit] = (),
    document_collection_factory: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory),
) -> tuple[DocumentCollection, LearningPolicyType | None]:
    docs_with_loaded_field = await documents.filter(lambda doc: doc.is_loaded(field_name)).with_fields([field_name])
    docs_without_loaded_field = documents - docs_with_loaded_field
    origin_to_docs = assign_to_origins(docs_without_loaded_field)
    preloaded_docs = await uniform_preload(origin_to_docs, field_name)
    docs_with_loaded_field += preloaded_docs

    field_load_count = 0
    loaded_docs = []
    if len(docs_without_loaded_field) > 0:
        adaptive_loader = AdaptiveLoader(
            origin_to_docs=origin_to_docs,
            preloaded_docs=docs_with_loaded_field,
            field=field_name,
            document_collection_factory=document_collection_factory,
            policy=LearningPolicy.BatchedThompsonSampling(
                batch_growth_factor=config_value(cfg_schema.relevance_judgement.batch_growth_factor),
                gaussian_variance=0.1,
                window_size=config_value(cfg_schema.relevance_judgement.window_size),
                initial_batch_size=config_value(cfg_schema.relevance_judgement.initial_batch_size),
                decay_factor=config_value(cfg_schema.relevance_judgement.decay_factor),
            ),
            batch_size=min(docs_quota, 50),
            load_quota=max(0, docs_quota - len(preloaded_docs)),
        )

        async for docs in adaptive_loader:
            loaded_docs += [doc for _, doc in docs]
            if shortcircuits and all(
                shortcircuit.should_break()
                for shortcircuit in [shortcircuit.accumulate(loaded_docs) for shortcircuit in shortcircuits]
            ):
                logger.info("Cap reached, breaking early.")
                break
            field_load_count += len(docs)
            if field_load_count >= docs_quota or not docs:
                break
        loaded_docs.extend([doc for _, doc in adaptive_loader.drain()])
        documents = DC.from_docs(loaded_docs, computed_fields=documents.computed_fields)
        return docs_with_loaded_field.merged(documents), adaptive_loader.policy
    else:
        logger.info("All documents already have the field loaded.")
        return docs_with_loaded_field, None


@configurable
async def uniform_preload(
    origin_to_docs: dict[str, DocumentCollection],
    field_name: DocumentFieldName,
    uniform_preload_size: int = ConfigValue(cfg_schema.relevance_judgement.uniform_preload_size),
) -> DocumentCollection:
    docs_for_preload = DC.merge(docs.take(uniform_preload_size) for docs in origin_to_docs.values())
    return await docs_for_preload.with_fields([field_name])


def assign_to_origins(
    docs_without_loaded_field: DocumentCollection, choose_best: bool = False
) -> dict[str, DocumentCollection]:
    if choose_best:
        return docs_without_loaded_field.group_by(extract_best_origin_query)
    else:
        return docs_without_loaded_field.multi_group_by(extract_origin_queries)


def extract_best_origin_query(doc: Document) -> str:
    if not doc.origins:
        return ""
    best_ranks = [float(min(o.ranks)) if o.ranks else float("inf") for o in doc.origins]
    total_best_rank = min(best_ranks)
    best_origins = [repr(origin) for origin, rank in zip(doc.origins, best_ranks) if rank == total_best_rank]
    if not best_origins:
        return ""
    if config_value(cfg_schema.force_deterministic):
        return best_origins[0]
    return random.choice(best_origins)


def extract_origin_queries(doc: Document) -> list[str]:
    if not doc.origins:
        return []
    return [repr(origin) for origin in doc.origins]


async def post_relevance_judgement_loading(
    documents: DocumentCollection, strategy: str, with_optimal: bool = False
) -> None:
    if config_value(cfg_schema.relevance_judgement.plot):
        documents_with_rj = documents.filter(lambda doc: doc.is_loaded("relevance_judgement"))
        plot_relevance_distribution(documents_with_rj, strategy=strategy)
        if with_optimal:
            documents_with_full_rj = await documents.with_fields(["relevance_judgement"])
            optimal_solution = find_optimal_solution(
                documents_with_full_rj, num_docs_to_evaluate=len(documents_with_rj)
            )
            plot_relevance_distribution(optimal_solution, strategy="Optimal")


def find_optimal_solution(
    documents: DocumentCollection,
    num_docs_to_evaluate: int,
) -> DocumentCollection:
    logger.info(f"Finding optimal solution for {num_docs_to_evaluate} documents.")
    origin_to_docs: dict[str, DocumentCollection] = assign_to_origins(documents)

    # Precompute best results for each possible number of evaluations per origin.
    origin_results = {}
    for origin, docs in origin_to_docs.items():
        max_relevance: list[float] = [0] * (len(docs) + 1)
        selected_docs: list[list[Document]] = [[] for _ in range(len(docs) + 1)]
        for i in range(1, len(docs) + 1):
            doc = docs.documents[i - 1]
            if doc.relevance_judgement is not None and doc.relevance_judgement.relevance in [1, 2, 3]:
                new_score = max_relevance[i - 1] + to_reward(doc, "relevance_judgement")
                if new_score > max_relevance[i - 1]:
                    max_relevance[i] = new_score
                    selected_docs[i] = selected_docs[i - 1] + [doc]
                else:
                    max_relevance[i] = max_relevance[i - 1]
                    selected_docs[i] = selected_docs[i - 1]
            else:
                max_relevance[i] = max_relevance[i - 1]
                selected_docs[i] = selected_docs[i - 1] + [doc]
        origin_results[origin] = (max_relevance, selected_docs)

    # Use dynamic programming to split num_docs_to_evaluate across origins
    dp: list[list[float]] = [[0] * (num_docs_to_evaluate + 1) for _ in range(len(origin_results) + 1)]
    document_picks: list[list[list[Document]]] = [
        [[] for _ in range(num_docs_to_evaluate + 1)] for _ in range(len(origin_results) + 1)
    ]
    origins = list(origin_results.keys())

    for i in range(1, len(origins) + 1):
        origin = origins[i - 1]
        max_relevance, selected_docs = origin_results[origin]
        for j in range(num_docs_to_evaluate + 1):
            for k in range(min(j + 1, len(max_relevance))):
                potential_new_score = dp[i - 1][j - k] + max_relevance[k]
                if potential_new_score > dp[i][j]:
                    dp[i][j] = potential_new_score
                    document_picks[i][j] = document_picks[i - 1][j - k] + selected_docs[k]

    optimal_docs = document_picks[-1][num_docs_to_evaluate]

    return DC.from_docs(optimal_docs)


def plot_relevance_distribution(documents: DocumentCollection, strategy: str, timestamp: str | None = None) -> Figure:
    if not config_value(cfg_schema.relevance_judgement.plot):
        return Figure()

    origin_to_docs = assign_to_origins(documents)

    # Prepare data for plotting
    origins: list[str] = sorted(list(origin_to_docs.keys()))
    relevance_levels = [3, 2, 1, 0]
    colors = ["#F0F0F0", "#F4B678", "#7CC674", "#38812F"]

    # Calculate total relevance counts across all origin queries
    total_relevance_counts: Counter = Counter()
    for origin_docs in origin_to_docs.values():
        total_relevance_counts.update(
            doc.relevance_judgement.relevance if doc.relevance_judgement is not None else -1
            for doc in origin_docs.documents
        )

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(21, 13))  # Increased figure height to accommodate total counts

    max_height = max(len(docs) for docs in origin_to_docs.values()) if origin_to_docs else 0

    for i, origin in enumerate(origins):
        docs: Sequence[Document] = origin_to_docs[origin].documents
        bottom = 0
        relevance_counts = Counter(
            doc.relevance_judgement.relevance if doc.relevance_judgement is not None else -1 for doc in docs
        )

        for doc in docs:
            relevance = doc.relevance_judgement.relevance if doc.relevance_judgement is not None else -1
            relevance = relevance if relevance in relevance_levels else 0
            ax.bar(i, 1, bottom=bottom, color=colors[relevance], width=0.8)
            bottom += 1

        # Add text annotations for relevance counts
        legend_text = "\n".join([f"R{level}: {relevance_counts[level]}" for level in relevance_levels])
        ax.text(i, len(docs) + 0.5, legend_text, ha="center", va="bottom", fontsize=10, wrap=True)

    # Add total relevance counts across all origin queries
    total_text = "Total counts: " + ", ".join(
        [f"R{level}: {total_relevance_counts[level]}" for level in relevance_levels]
    )
    fig.text(0.5, 0.95, total_text, ha="center", va="top", fontsize=12)

    # Customize the plot
    ts_title = timestamp or datetime.now().strftime("%y%m%d-%Hh%Mm%Ss")
    ax.set_title(f"Document Relevance Distribution by Origin [{ts_title}]\n{strategy}", fontsize=16, pad=30)
    ax.set_xlabel("Origin", fontsize=12)
    ax.set_ylabel("Count of Documents", fontsize=12)
    ax.legend(
        [Rectangle((0, 0), 1, 1, fc=colors[level]) for level in relevance_levels],
        [f"Relevance {level}" for level in relevance_levels],
        title="Relevance Level",
        fontsize=10,
        title_fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )

    # Wrap x-axis labels
    wrapped_labels = [textwrap.fill(label, width=12) for label in origins]
    ax.set_xticks(range(len(origins)))
    ax.set_xticklabels(wrapped_labels, fontsize=8)

    # Adjust y-axis to accommodate the text annotations
    ax.set_ylim(0, max_height + 5)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for the total counts text

    # plt.show() # uncomment to display the plot in a window

    # save the plot as a png file to __file__ directory /plots folder with the timestamp as the filename
    folder = Path(__file__).parent / "plots"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f"{ts_title}.png", dpi=300)

    return fig
