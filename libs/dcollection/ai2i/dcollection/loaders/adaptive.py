from __future__ import annotations

import sys
from typing import AsyncIterator, Iterable, Iterator, Mapping, Protocol, Sequence, cast

from mabwiser.mab import MAB, LearningPolicyType
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ai2i.dcollection.factory import DocumentCollectionFactory
from ai2i.dcollection.interface.collection import Document, DocumentCollection
from ai2i.dcollection.interface.document import DocumentFieldName

type Origin = str
type OriginDocuments = Sequence[tuple[Origin, Document]]


class FieldLoaderWithReward(Protocol):
    async def load(self, doc: Document, field: str) -> float: ...


class AdaptiveLoader(BaseModel, AsyncIterator[OriginDocuments]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    origin_to_docs: dict[Origin, DocumentCollection | Iterable[Document]]
    preloaded_docs: DocumentCollection = Field()
    field: DocumentFieldName
    document_collection_factory: DocumentCollectionFactory = Field()
    batch_size: int = Field(1)
    load_quota: int
    policy: LearningPolicyType
    _used_quota: int = PrivateAttr(default=0)

    _origin_to_iterators: dict[Origin, Iterator[Document]] = PrivateAttr()

    def __init__(
        self,
        origin_to_docs: Mapping[Origin, DocumentCollection | Iterable[Document]],
        preloaded_docs: DocumentCollection,
        document_collection_factory: DocumentCollectionFactory,
        field: DocumentFieldName,
        policy: LearningPolicyType,
        batch_size: int = 1,
        load_quota: int = sys.maxsize,
    ) -> None:
        super().__init__(
            origin_to_docs=origin_to_docs,
            preloaded_docs=preloaded_docs or document_collection_factory.empty(),
            document_collection_factory=document_collection_factory,
            field=field,
            batch_size=batch_size,
            load_quota=load_quota,
            policy=policy,
        )
        self._initialize_iterators()
        self._initialize_mab()

    def _initialize_iterators(self) -> None:
        self._origin_to_iterators = {
            origin: iter(docs.documents) if isinstance(docs, DocumentCollection) else iter(docs)
            for origin, docs in self.origin_to_docs.items()
        }

    def _initialize_mab(self) -> None:
        origins = list(self.origin_to_docs.keys())
        self._mab = MAB(origins, self.policy)
        decisions, rewards = [], []
        for preloaded_doc in self.preloaded_docs.documents:
            if preloaded_doc.origins is not None:
                for origin in preloaded_doc.origins:
                    decisions.append(origin)
                    rewards.append(to_reward(preloaded_doc, self.field))
        self._mab.fit(decisions, rewards)

    def __aiter__(self) -> AsyncIterator[OriginDocuments]:
        return self

    async def __anext__(self) -> OriginDocuments:
        batch = await self._get_next_batch()
        if not batch:
            raise StopAsyncIteration
        return await self._load_batch(batch)

    async def _get_next_batch(self) -> OriginDocuments:
        batch: OriginDocuments = []
        batch_size = await self._get_next_batch_size()
        while len(batch) < batch_size and self._mab.arms:
            origin = cast(Origin, self._mab.predict())
            document = await self._get_next_unloaded_document(origin)
            if document:
                if not document.is_loaded(self.field):
                    self._used_quota += 1
                batch.append((origin, document))
            else:
                self._mab.remove_arm(origin)
        return batch

    async def _get_next_batch_size(self) -> int:
        mab_imp = self._mab._imp
        batch_size: int = (
            mab_imp.get_current_batch_size()  # type: ignore
            if hasattr(mab_imp, "get_current_batch_size")
            else self.batch_size
        )
        return min(batch_size, self.load_quota - self._used_quota)

    async def _get_next_unloaded_document(self, origin: Origin) -> Document | None:
        iterator = self._origin_to_iterators[origin]
        try:
            doc = next(doc for doc in iterator)
            return doc
        except StopIteration:
            return None

    async def _load_batch(self, batch: OriginDocuments) -> OriginDocuments:
        origins, documents = zip(*batch)
        loaded_docs: Sequence[Document] = (
            await self.document_collection_factory.from_docs(
                documents, computed_fields=self.preloaded_docs.computed_fields
            ).with_fields([self.field])
        ).documents

        loaded_doc_by_id = {doc.corpus_id: doc for doc in loaded_docs}
        reordered_loaded_docs = [loaded_doc_by_id[original_doc.corpus_id] for original_doc in documents]

        rewards = [to_reward(doc, self.field) for doc in reordered_loaded_docs]

        if self._mab.arms:
            self._mab.partial_fit(list(origins), rewards)

        return list(zip(origins, reordered_loaded_docs))

    def drain(self) -> OriginDocuments:
        unloaded_docs = []
        for origin, iterator in self._origin_to_iterators.items():
            for doc in iterator:
                unloaded_docs.append((origin, doc))
        return unloaded_docs


def to_reward(doc: Document, field: str) -> float:
    return ((2 ** float(doc[field] or 0)) / 8.0) - 0.125
