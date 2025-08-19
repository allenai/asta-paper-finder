from __future__ import annotations

from collections import defaultdict
from random import shuffle
from typing import Sequence

from ai2i.dcollection.interface.collection import Document
from ai2i.dcollection.interface.document import SampleMethod


def sample(documents: Sequence[Document], n: int, method: SampleMethod) -> Sequence[Document]:
    if not documents:
        return []

    def get_stratified_doc_sample(documents: Sequence[Document], strat_n: int) -> Sequence[Document]:
        relevance_groups = defaultdict(list)
        for doc in documents:
            relevance_groups[doc.relevance_judgement.relevance if doc.relevance_judgement is not None else -1].append(
                doc
            )

        stratified_docs = []
        taken_docs = 0
        for relevance_value in sorted(relevance_groups.keys(), reverse=True):
            group_docs = relevance_groups[relevance_value]
            shuffle(group_docs)  # Randomize within each relevance level.
            num_docs_to_take = min(strat_n - taken_docs, len(group_docs))
            stratified_docs.extend(group_docs[:num_docs_to_take])
            taken_docs += num_docs_to_take
            if taken_docs >= strat_n:
                break
        return sorted(stratified_docs, key=lambda d: d.corpus_id)

    def get_relevance_origin_sample(documents: Sequence[Document], sample_n: int) -> Sequence[Document]:
        origin_groups: dict[str, list[Document]] = defaultdict(list[Document])
        for doc in documents:
            origin_groups[(repr(max(doc.origins, key=doc.origins.count)) if doc.origins else "")].append(doc)

        relevance_index_doc_tuples = []
        for group_docs in origin_groups.values():
            # This assumes original dense retrieval results ranking order is preserved here.
            for reversed_group_index, doc in enumerate(reversed(group_docs)):
                relevance_index_doc_tuples.append(
                    (
                        doc.relevance_judgement.relevance if doc.relevance_judgement is not None else -1,
                        reversed_group_index,
                        doc,
                    )
                )

        # Sort by relevance (descending) and reversed_group_index (descending)
        relevance_index_doc_tuples.sort(
            key=lambda relevance_index_doc: (
                -relevance_index_doc[0],
                -relevance_index_doc[1],
                relevance_index_doc[2].corpus_id,
            )
        )

        return [doc for _, _, doc in relevance_index_doc_tuples[:sample_n]]

    if not isinstance(documents[0], Document):
        raise ValueError("Sampling is only supported for Document instances.")

    sample_docs: Sequence[Document]

    match method:
        case "random":
            shuffled_docs = list(documents)
            shuffle(shuffled_docs)
            sample_docs = shuffled_docs[:n]
        case "top_relevance":
            sample_docs = sorted(
                documents,
                key=lambda doc: doc.relevance_judgement.relevance if doc.relevance_judgement is not None else -1,
                reverse=True,
            )[:n]
        case "bottom_origin_rank_stratified_relevance":
            sample_docs = get_relevance_origin_sample(documents, n)
        case "random_stratified_relevance":
            sample_docs = get_stratified_doc_sample(documents, n)
        case _:
            raise ValueError(f"Unsupported sampling method: {method}")

    return sample_docs
