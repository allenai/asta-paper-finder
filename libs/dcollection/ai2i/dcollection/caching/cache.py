from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Sequence

from aiocache import Cache
from aiocache.serializers import PickleSerializer

from ai2i.common.utils.asyncio import custom_gather
from ai2i.common.utils.value import ValueNotSet
from ai2i.dcollection.data_access_context import (
    DynamicallyLoadedEntity,
    EntityId,
    FieldRequirements,
    QueryFnSansContext,
)
from ai2i.dcollection.interface.collection import DocLoadingError


class SubsetCache:
    def __init__(
        self,
        ttl: int,
        is_enabled: bool = True,
        force_deterministic: bool = False,
    ) -> None:
        self.cache = Cache(Cache.MEMORY, serializer=PickleSerializer(), namespace="cache")
        self.ttl = ttl
        self.is_enabled = is_enabled
        self.force_deterministic = force_deterministic

    def enabled(self) -> bool:
        return self.is_enabled

    async def fetch_async_data[DFN: str](
        self,
        query_fn: QueryFnSansContext[DFN],
        entities: Sequence[DynamicallyLoadedEntity[DFN]],
        fields: Sequence[FieldRequirements[DFN]],
    ) -> list[DynamicallyLoadedEntity[DFN]]:
        if not entities or not fields:
            return list(entities)

        # Fetch data from cache.
        keys_lookup = CacheKeysLookup.build(entities, fields)
        cached_values: list[CacheValue] = await self.cache.multi_get(keys_lookup.cacheable_keys)
        cache_result = self._process_cache_results(entities, fields, keys_lookup, cached_values)

        # Fetch missing data from the query function.
        query_results: dict[EntityId, DynamicallyLoadedEntity[DFN]] = {}
        if cache_result.entities_to_query:
            query_results = await self._fetch_missing_data(
                query_fn, cache_result.entities_to_query, cache_result.fields_to_query
            )

            if query_results:
                # Cache the fetched data.
                await self._cache_fetched_data(
                    list(query_results.values()),
                    cache_result.fields_to_query,
                    keys_lookup,
                )

        # Apply the fetched data to the entities.
        return self._apply_all_values(
            entities=entities,
            fields=cache_result.fields_to_query,
            cache_hits=cache_result.cache_hits,
            query_results=query_results,
        )

    def _process_cache_results[DFN: str](
        self,
        entities: Sequence[DynamicallyLoadedEntity[DFN]],
        fields: Sequence[FieldRequirements[DFN]],
        keys_lookup: CacheKeysLookup[DFN],
        cached_values: list[CacheValue],
    ) -> CacheResult[DFN]:
        cache_hits: dict[EntityId, dict[DFN, FieldValue]] = {}
        entities_to_query: set[DynamicallyLoadedEntity[DFN]] = set()

        cached_values_by_key = dict(
            zip(
                ((key[0]["entity_id"], key[1]) for key in keys_lookup.cache_keys.values()),
                cached_values,
            )
        )

        # Collect entities to query.
        for entity in entities:
            field_values = {}
            for field_req in fields:
                cached_value = cached_values_by_key.get((entity.entity_id, field_req.field))
                if cached_value is not None:
                    field_values[field_req.field] = FieldValue(cached_value)

            if len(field_values) < len(fields):
                entities_to_query.add(entity)
            if field_values:
                cache_hits[entity.entity_id] = field_values

        # Collect fields to query.
        fields_to_query = set(
            {
                field.field
                for entity in sorted(entities_to_query, key=lambda e: e.entity_id)
                for field in fields
                if cached_values_by_key.get((entity.entity_id, field.field)) is None
            }
        )

        return CacheResult(cache_hits, entities_to_query, fields_to_query)

    def _apply_all_values[DFN: str](
        self,
        entities: Sequence[DynamicallyLoadedEntity[DFN]],
        fields: set[DFN],
        cache_hits: dict[EntityId, dict[DFN, FieldValue]],
        query_results: dict[EntityId, DynamicallyLoadedEntity[DFN]],
    ) -> list[DynamicallyLoadedEntity[DFN]]:
        result_entities = list(entities)

        for entity in result_entities:
            if entity.entity_id in cache_hits:
                for field_name, field_value in cache_hits[entity.entity_id].items():
                    setattr(entity, field_name, field_value.actual_value)

            if entity.entity_id in query_results:
                queried_entity = query_results[entity.entity_id]
                for field in fields:
                    if hasattr(queried_entity, field):
                        query_result_value = getattr(queried_entity, field)
                        if query_result_value is not None or not entity.is_loaded(field):
                            setattr(entity, field, query_result_value)

        return result_entities

    async def _fetch_missing_data[DFN: str](
        self,
        query_fn: QueryFnSansContext[DFN],
        entities_to_query: set[DynamicallyLoadedEntity[DFN]],
        fields_to_query: set[DFN],
    ) -> dict[EntityId, DynamicallyLoadedEntity[DFN]]:
        if not entities_to_query or not fields_to_query:
            return {}

        sorted_entities = sorted(entities_to_query, key=lambda e: e.entity_id)
        sorted_fields = sorted(fields_to_query)
        fetched = await query_fn(sorted_entities, list(sorted_fields))
        return {e.entity_id: e for e in fetched}

    async def _cache_fetched_data[DFN: str](
        self,
        fetched_entities: list[DynamicallyLoadedEntity[DFN]],
        fields_to_query: set[DFN],
        keys_lookup: CacheKeysLookup[DFN],
    ) -> None:
        sorted_entities = sorted(fetched_entities, key=lambda e: e.entity_id)

        await custom_gather(
            self._cache_assigned_values(sorted_entities, fields_to_query, keys_lookup, self.ttl),
            self._cache_none_values(sorted_entities, fields_to_query, keys_lookup, self.ttl),
            force_deterministic=self.force_deterministic,
        )

    async def _cache_assigned_values[DFN: str](
        self,
        sorted_entities: list[DynamicallyLoadedEntity[DFN]],
        fields_to_query: set[DFN],
        keys_lookup: CacheKeysLookup[DFN],
        ttl: int,
    ) -> None:
        to_cache = [
            (
                keys_lookup.lookup_key(entity.entity_id, field_name),
                getattr(entity, field_name),
            )
            for field_name in sorted(fields_to_query)
            for entity in sorted_entities
            if not isinstance(getattr(entity, field_name), DocLoadingError) and getattr(entity, field_name) is not None
        ]
        if to_cache:
            await self.cache.multi_set(to_cache, ttl=ttl)

    async def _cache_none_values[DFN: str](
        self,
        sorted_entities: list[DynamicallyLoadedEntity[DFN]],
        fields_to_query: set[DFN],
        keys_lookup: CacheKeysLookup[DFN],
        ttl: int,
    ) -> None:
        none_keys = [
            keys_lookup.lookup_key(entity.entity_id, field_name)
            for field_name in sorted(fields_to_query)
            for entity in sorted_entities
            if not isinstance(getattr(entity, field_name), DocLoadingError) and getattr(entity, field_name) is None
        ]
        if none_keys:
            existing_values = await self.cache.multi_get(none_keys)
            none_values_to_cache = [
                (key, ValueNotSet) for key, value in zip(none_keys, existing_values) if value is None
            ]
            if none_values_to_cache:
                await self.cache.multi_set(none_values_to_cache, ttl=ttl)

    async def put[DFN: str](
        self,
        entities: Sequence[DynamicallyLoadedEntity[DFN]],
        fields: Sequence[FieldRequirements[DFN]],
    ) -> None:
        """Store entity field values in the cache."""
        if not self.is_enabled:
            return

        keys_lookup = CacheKeysLookup.build(entities, fields)
        to_cache: list[tuple[CacheKey, CacheValue]] = []

        for entity in entities:
            for field_req in fields:
                if hasattr(entity, field_req.field):
                    value = getattr(entity, field_req.field)
                    if value is not None:
                        cache_key = keys_lookup.lookup_key(entity.entity_id, field_req.field)
                        to_cache.append((cache_key, value))

        if to_cache:
            await self.cache.multi_set(to_cache, ttl=self.ttl)

    async def clear(self) -> None:
        """Clear all cached data."""
        await self.cache.clear()


@dataclass(frozen=True)
class FieldValue:
    value: CacheValue

    @property
    def actual_value(self) -> CacheValue:
        return None if self.value == ValueNotSet else self.value


class CacheResult[DFN: str](NamedTuple):
    cache_hits: dict[EntityId, dict[DFN, FieldValue]]
    entities_to_query: set[DynamicallyLoadedEntity[DFN]]
    fields_to_query: set[DFN]


type CacheKey[DFN: str] = tuple[dict[EntityId, Any], DFN]
type CacheValue = Any
type CacheKeyTuple[DFN: str] = tuple[EntityId, DFN]


@dataclass
class CacheKeysLookup[DFN: str]:
    # This additional lookup is necessary because CacheKey is not (efficiently) hashable.
    cache_keys: dict[CacheKeyTuple[DFN], CacheKey[DFN]] = field(default_factory=dict)

    @classmethod
    def build[E: DynamicallyLoadedEntity](
        cls, entities: Sequence[E], fields: Sequence[FieldRequirements[DFN]]
    ) -> CacheKeysLookup[DFN]:
        """Build a cache keys lookup for the given entities and fields."""
        cache_keys: dict[CacheKeyTuple[DFN], CacheKey[DFN]] = {}

        for entity in entities:
            entity_id = entity.entity_id
            for field_req in fields:
                required_fields = (
                    entity.model_dump(include=set(str(rf) for rf in field_req.required_fields))
                    if field_req.required_fields
                    else {}
                )

                computation_id = entity.get_dynamic_field_computation_id(field_req.field)
                cache_dict = {
                    "entity_id": entity_id,
                    **required_fields,
                }
                if computation_id is not None:
                    cache_dict["computation_hash"] = hash(computation_id)

                cache_keys[(entity_id, field_req.field)] = (
                    cache_dict,
                    field_req.field,
                )

        return cls(cache_keys=cache_keys)

    @property
    def cacheable_keys(self) -> list[CacheKey[DFN]]:
        return list(self.cache_keys.values())

    def lookup_key(self, entity_id: EntityId, field_name: DFN) -> CacheKey[DFN]:
        return self.cache_keys[(entity_id, field_name)]
