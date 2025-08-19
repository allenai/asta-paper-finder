from __future__ import annotations

import logging
from collections import Counter, defaultdict
from functools import partial
from typing import (
    Any,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    cast,
)

import pandas as pd
from pydantic import field_serializer, field_validator, model_validator
from pydantic.fields import FieldInfo

from ai2i.common.utils.asyncio import custom_gather
from ai2i.common.utils.data_struct import SortedSet
from ai2i.dcollection.computed_field import (
    AggTransformComputedField,
    AssignedField,
    BatchComputedField,
    ComputedField,
)
from ai2i.dcollection.data_access_context import (
    ComputationId,
    DocumentCollectionContext,
    DynamicallyLoadedEntity,
    DynamicField,
    FieldRequirements,
)
from ai2i.dcollection.document import PaperFinderDocument
from ai2i.dcollection.interface.collection import (
    BaseComputedField,
    Document,
    DocumentCollection,
    DocumentCollectionSortDef,
    DocumentEnumProjector,
    DocumentFieldLoader,
    DocumentPredicate,
    DocumentProjector,
    QueryFn,
    TakeFirst,
)
from ai2i.dcollection.interface.document import (
    CorpusId,
    DocumentFieldName,
    SampleMethod,
)
from ai2i.dcollection.sampling import sample

logger = logging.getLogger(__name__)


class PaperFinderDocumentCollection(DocumentCollection):
    def __repr__(self) -> str:
        return (
            f"DocumentCollection({len(self.documents)} documents, computed_fields={self.computed_fields}); "
            f"Preview: {self.documents[:1]}"
        )

    @field_serializer("documents")
    def serialize_documents(self, documents: Sequence[Document]) -> list[Document]:
        return list(documents)

    @field_validator("documents", mode="after")
    @classmethod
    def dedup(cls, documents: Sequence[Document]) -> Sequence[Document]:
        duplicates = [item for item, count in Counter([d.corpus_id for d in documents]).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate id(s) found in a collection: {duplicates}")
        return documents

    @model_validator(mode="after")
    def assign_dynamic_fields(self) -> PaperFinderDocumentCollection:
        for doc in self.documents:
            doc.dynamic_fields = self.computed_fields
        return self

    async def with_fields(
        self,
        fields: Sequence[DocumentFieldName | BaseComputedField[DocumentFieldName, Any]],
    ) -> PaperFinderDocumentCollection:
        self._update_computed_fields(fields)
        field_names_to_load = [field.field_name if isinstance(field, BaseComputedField) else field for field in fields]
        docs_with_loaded_fields = await self.load_many(self.documents, field_names_to_load)
        return PaperFinderDocumentCollection(
            documents=docs_with_loaded_fields, computed_fields=self.computed_fields, factory=self.factory
        )

    async def load_many(
        self, entities: Sequence[Document], field_names: Sequence[DocumentFieldName]
    ) -> Sequence[Document]:
        if len(entities) == 0:
            return entities
        loader_funcs = self.loaders_for_fields(field_names)
        loading_tasks = []
        fields_groups: list[list[FieldRequirements[DocumentFieldName]]] = []
        entities_by_id = {e.entity_id: e for e in entities}
        for loader_func, fields_to_load in loader_funcs.items():
            fields_groups.append(fields_to_load)
            required_fields: list[DocumentFieldName] = [rf for f in fields_to_load for rf in f.required_fields]
            entities_with_loaded_requirements = await self.load_many(entities, required_fields)
            entities_limited_to_loaded_requirements = cast(
                list[DynamicallyLoadedEntity[DocumentFieldName]],
                [e.clone_partial(required_fields) for e in entities_with_loaded_requirements],
            )

            cache = self.factory.cache()
            query_func = cast(QueryFn[DocumentFieldName], loader_func)
            if cache.enabled and not any(f.skip_cache for f in fields_to_load):
                query_func_with_context = partial(query_func, context=self.factory.context())
                cached_loader_func = partial(cache.fetch_async_data, query_func_with_context)
                loading_tasks.append(cached_loader_func(entities_limited_to_loaded_requirements, fields_to_load))
            else:
                loading_tasks.append(
                    query_func(
                        entities_limited_to_loaded_requirements,  # type: ignore
                        [f.field for f in fields_to_load],
                        context=self.factory.context(),
                    )
                )

        loaded_entities_results = await custom_gather(
            *loading_tasks, force_deterministic=self.factory.context().force_deterministic
        )

        loaded_entities_results_for_fields = list(zip(fields_groups, loaded_entities_results))

        for loaded_fields, loaded_entity_result in loaded_entities_results_for_fields:
            for loaded_entity in loaded_entity_result:
                entity = entities_by_id[loaded_entity.entity_id]
                if entity:
                    entity.assign_loaded_values([lf.field for lf in loaded_fields], [loaded_entity])

        return list(entities)

    def loaders_for_fields(
        self,
        field_names: Sequence[DocumentFieldName],
    ) -> FieldLoadersToRequirements:
        loader_funcs = dict[DocumentFieldLoader, list[FieldRequirements[DocumentFieldName]]]()
        for field_name in field_names:
            field = self._get_dynamic_field(field_name)
            if isinstance(field, DynamicField):
                if field:
                    if field.loading_functions:
                        for loader_func in field.loading_functions:
                            if loader_func not in loader_funcs:
                                loader_funcs[loader_func] = []
                            loader_funcs[loader_func].append(
                                FieldRequirements[str](
                                    field=field_name,
                                    required_fields=cast(Sequence[str], field.required_fields),
                                    skip_cache=not field.cache,
                                )
                            )
                    elif field.mandatory_loader:
                        raise ValueError(
                            f"Dynamic computed field {field_name} has no mandatory loading functions defined."
                        )
        return loader_funcs

    def _get_dynamic_field(self, field_name: DocumentFieldName) -> DynamicField | FieldInfo | None:
        return self.computed_fields.get(
            field_name,
            PaperFinderDocument.get_predefined_dynamic_fields().get(field_name),
        )

    def update_computed_fields(
        self, fields: Sequence[DocumentFieldName | BaseComputedField]
    ) -> PaperFinderDocumentCollection:
        collection = PaperFinderDocumentCollection(
            documents=self.documents, computed_fields=self.computed_fields, factory=self.factory
        )
        collection._update_computed_fields(fields)
        return collection

    def _update_computed_fields(self, fields: Sequence[DocumentFieldName | BaseComputedField]) -> None:
        computed_fields = [field for field in fields if isinstance(field, BaseComputedField)]
        if computed_fields:
            for computed_field in computed_fields:
                self._update_computed_field(computed_field)
                if isinstance(computed_field, AssignedField):
                    computed_field.values_to_docs(self.documents)

    def _update_computed_field(self, computed_field: BaseComputedField) -> None:
        computation_loader: DocumentFieldLoader | None
        should_cache_field: bool
        required_fields: list[DocumentFieldName]
        computation_id: ComputationId
        match computed_field:
            case ComputedField():
                # This wraps a single doc->value computation function into a loader
                # function, which conforms with the batch loader API, and thus,
                # batch-provides the required fields to the loader.
                async def computation_based_loader(
                    entities: Sequence[Document],
                    fields: Sequence[DocumentFieldName],
                    context: DocumentCollectionContext,
                ) -> Sequence[Document]:
                    for entity in entities:
                        entity[computed_field.field_name] = computed_field.computation(entity)
                    return list(entities)

                computation_loader = computation_based_loader
                required_fields = computed_field.required_fields
                should_cache_field = computed_field.use_cache
                computation_id = computed_field.computation_id
            case BatchComputedField() | AggTransformComputedField():

                async def batch_computation_based_loader(
                    entities: Sequence[Document],
                    fields: Sequence[DocumentFieldName],
                    context: DocumentCollectionContext,
                ) -> Sequence[Document]:
                    batch_results = await computed_field.computation(entities)
                    for entity, batch_result in zip(entities, batch_results):
                        entity[computed_field.field_name] = batch_result
                    return list(entities)

                computation_loader = batch_computation_based_loader
                required_fields = computed_field.required_fields
                should_cache_field = computed_field.use_cache
                computation_id = computed_field.computation_id
            case AssignedField():
                computation_loader = None
                required_fields = computed_field.required_fields
                should_cache_field = computed_field.use_cache
                computation_id = computed_field.computation_id
            case _:
                raise ValueError(f"Invalid computed field type: {computed_field}")
        dynamic_computed_field = DynamicField(
            loaders=[computation_loader] if computation_loader else None,
            computation_id=computation_id,
            fuse=TakeFirst(),
            required_fields=required_fields,
            cache=should_cache_field,
            extra=True,
        )
        self.computed_fields[computed_field.field_name] = dynamic_computed_field

    def merged(self, *collections: DocumentCollection) -> DocumentCollection:
        if not collections:
            return self

        docs_by_corpus_id = keyed_by_corpus_id(list(self.documents))

        merged_computed_fields = defaultdict(SortedSet)
        for cf, value in self.computed_fields.items():
            merged_computed_fields[cf].add(value)

        for collection in collections:
            for cf, value in collection.computed_fields.items():
                merged_computed_fields[cf].add(value)

            if any(len(values) > 1 for values in merged_computed_fields.values()):
                raise ValueError(
                    f"Cannot merge collections with overriding computed fields. "
                    f"Self computed fields: {self.computed_fields}, "
                    f"Other collection computed fields: {collection.computed_fields}"
                )

            for document in collection.documents:
                if document.corpus_id in docs_by_corpus_id:
                    docs_by_corpus_id[document.corpus_id] = docs_by_corpus_id[document.corpus_id].fuse(document)
                else:
                    doc_to_add = document.clone_partial()
                    docs_by_corpus_id[document.corpus_id] = doc_to_add

        return PaperFinderDocumentCollection(
            documents=list(docs_by_corpus_id.values()),
            computed_fields={k: next(iter(v)) for k, v in merged_computed_fields.items()},
            factory=self.factory,
        )

    def __add__(self, other: DocumentCollection) -> DocumentCollection:
        return self.merged(other)

    def iter(self) -> Iterator[Document]:
        return iter(self.documents)

    def take(self, n: int) -> DocumentCollection:
        return PaperFinderDocumentCollection(
            documents=self.documents[:n], computed_fields=self.computed_fields, factory=self.factory
        )

    def filter(self, filter_fn: DocumentPredicate) -> DocumentCollection:
        return PaperFinderDocumentCollection(
            documents=[doc for doc in self.documents if filter_fn(doc)],
            computed_fields=self.computed_fields,
            factory=self.factory,
        )

    def __sub__(self, other: DocumentCollection) -> DocumentCollection:
        return self.subtract(other)

    def subtract(self, other: DocumentCollection) -> DocumentCollection:
        other_corpus_ids = {doc.corpus_id for doc in other.documents}
        return self.filter(lambda doc: doc.corpus_id not in other_corpus_ids)

    def map(self, map_fn: DocumentProjector[Document]) -> DocumentCollection:
        return PaperFinderDocumentCollection(
            documents=[map_fn(doc) for doc in self.documents],
            computed_fields=self.computed_fields,
            factory=self.factory,
        )

    def map_enumerate(self, map_fn: DocumentEnumProjector[Document]) -> DocumentCollection:
        return PaperFinderDocumentCollection(
            documents=[map_fn(i, doc) for i, doc in enumerate(self.documents)],
            computed_fields=self.computed_fields,
            factory=self.factory,
        )

    def project[V](self, map_fn: DocumentProjector[V]) -> list[V]:
        return [map_fn(doc) for doc in self.documents]

    def flat_project[V](self, map_fn: DocumentProjector[Sequence[V]]) -> list[V]:
        return [value for values in self.project(map_fn) for value in values]

    def __len__(self) -> int:
        return len(self.documents)

    def __bool__(self) -> bool:
        return bool(self.documents)

    def group_by[V](self, group_fn: DocumentProjector[V]) -> dict[V, DocumentCollection]:
        groups = defaultdict(list)
        for doc in self.documents:
            groups[group_fn(doc)].append(doc)
        return {
            key: PaperFinderDocumentCollection(
                documents=group,
                computed_fields=self.computed_fields,
                factory=self.factory,
            )
            for key, group in groups.items()
        }

    def multi_group_by[V](self, group_fn: DocumentProjector[Sequence[V]]) -> dict[V, DocumentCollection]:
        groups = defaultdict(list)
        for doc in self.documents:
            group_values = group_fn(doc)
            for value in group_values:
                if doc not in groups[value]:
                    groups[value].append(doc)

        return {
            key: PaperFinderDocumentCollection(
                documents=group,
                computed_fields=self.computed_fields,
                factory=self.factory,
            )
            for key, group in groups.items()
        }

    def sorted(self, sort_definitions: Sequence[DocumentCollectionSortDef]) -> DocumentCollection:
        documents = list(self.documents)
        # Apply each sort definition in reverse order (since each sort is stable, it will keep previous sorts intact).
        for sort_def in reversed(sort_definitions):
            documents.sort(
                key=lambda x: getattr(x, sort_def.field_name),
                reverse=sort_def.order == "desc",
            )
        return PaperFinderDocumentCollection(
            documents=documents,
            computed_fields=self.computed_fields,
            factory=self.factory,
        )

    def sample(self, n: int, method: SampleMethod) -> DocumentCollection:
        sampled_docs = sample(self.documents, n, method)
        return PaperFinderDocumentCollection(
            documents=sampled_docs,
            computed_fields=self.computed_fields,
            factory=self.factory,
        )

    def to_dataframe(
        self,
        fields: list[str],
        handle_missing_fields: Literal["raise", "fill", "skip_doc"] = "raise",
    ) -> pd.DataFrame:
        data: list[dict] = []
        for doc in self.documents:
            doc_items: dict[str, Any] = {}
            skip_doc = False
            for field in fields:
                if not doc.is_loaded(field):
                    match handle_missing_fields:
                        case "fill":
                            doc_items[field] = None
                        case "skip_doc":
                            skip_doc = True
                            break
                        case "raise":
                            raise ValueError(f"Field {field} not loaded in entity_id={doc.corpus_id}.")
                        case _:
                            raise ValueError(f"Invalid handle_missing_fields: {handle_missing_fields}")
                else:
                    doc_items[field] = doc[field]
            if not skip_doc:
                data.append(doc_items)

        return pd.DataFrame(data)

    def to_debug_dataframe(
        self,
    ) -> pd.DataFrame:
        data: list[dict] = []
        for doc in self.documents:
            doc_fields = doc.get_loaded_fields()
            doc_items = {}
            for field in doc_fields:
                doc_items[field] = doc[field]
            data.append(doc_items)

        return pd.DataFrame(data)

    def to_field_requirements(
        self, field_names: Sequence[DocumentFieldName]
    ) -> Sequence[FieldRequirements[DocumentFieldName]]:
        field_requirements: list[FieldRequirements[DocumentFieldName]] = []
        for field_name in field_names:
            field = self._get_dynamic_field(field_name)
            if isinstance(field, DynamicField):
                field_requirements.append(
                    cast(
                        FieldRequirements[DocumentFieldName],
                        FieldRequirements(field_name, field.required_fields),
                    )
                )
        return field_requirements


type FieldLoadersToRequirements = Mapping[DocumentFieldLoader, list[FieldRequirements[DocumentFieldName]]]


def keyed_by_corpus_id(documents: Sequence[Document]) -> dict[CorpusId, Document]:
    return {doc.corpus_id: doc for doc in documents}
