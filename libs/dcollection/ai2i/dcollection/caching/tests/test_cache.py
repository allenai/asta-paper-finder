from __future__ import annotations

from typing import Any, AsyncIterator, ClassVar, List, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import Field

from ai2i.common.utils.value import ValueNotSet
from ai2i.dcollection import DocumentCollectionFactory
from ai2i.dcollection.caching.cache import SubsetCache
from ai2i.dcollection.data_access_context import (
    ComputationId,
    DynamicallyLoadedEntity,
    EntityId,
    FieldRequirements,
    SubsetCacheInterface,
)
from ai2i.dcollection.interface.collection import DocLoadingError


class SampleEntity(DynamicallyLoadedEntity[str]):
    id: str = Field()
    data: str | DocLoadingError | None = Field(default=None)
    more_data: str | None = Field(default=None)
    extra_data: str | DocLoadingError | None = Field(default=None)
    dynamic_fields_computation_ids: ClassVar[dict[str, str]] = {}

    @property
    def entity_id(self) -> EntityId:
        return self.id

    @entity_id.setter
    def entity_id(self, value: EntityId) -> None:
        self.id = value

    def is_loaded(self, field_name: str) -> bool:
        return False

    def clear_loaded_field(self, field_name: str) -> None: ...

    def get_dynamic_field_computation_id(self, field_name: str) -> ComputationId | None:
        return self.dynamic_fields_computation_ids.get(field_name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SampleEntity):
            return False
        return self.entity_id == other.entity_id

    def __hash__(self) -> int:
        return hash(self.entity_id)


@pytest.fixture
def factory() -> DocumentCollectionFactory:
    return DocumentCollectionFactory()


@pytest.fixture(scope="function")
async def subset_cache(factory: DocumentCollectionFactory) -> AsyncIterator[SubsetCacheInterface]:
    cache = factory.cache()
    yield cache
    await cache.clear()


@pytest.fixture
def entities() -> List[SampleEntity]:
    return [
        SampleEntity(id="1", data="data1", more_data="more_data1", extra_data="extra_data1"),
        SampleEntity(id="2", data="data2", more_data="more_data2", extra_data="extra_data2"),
        SampleEntity(id="3", data="data3", more_data="more_data3", extra_data="extra_data3"),
    ]


def to_field_reqs(fields: list[str]) -> list[FieldRequirements]:
    return [FieldRequirements(field=field, required_fields=[]) for field in fields]


@pytest.mark.asyncio
async def test_fetch_async_data_partial_cache_hit(entities: List[SampleEntity], subset_cache: SubsetCache) -> None:
    with (
        patch.object(subset_cache.cache, "multi_get", wraps=subset_cache.cache.multi_get) as mock_multi_get,
        patch.object(subset_cache.cache, "multi_set", wraps=subset_cache.cache.multi_set) as mock_multi_set,
    ):
        query_fn = AsyncMock(
            return_value=[
                SampleEntity(id="1", data="new_data1", extra_data="new_extra_data1"),
                SampleEntity(id="2", data="new_data2", extra_data="new_extra_data2"),
            ]
        )

        result = cast(
            list[SampleEntity],
            await subset_cache.fetch_async_data(query_fn, entities, to_field_reqs(["data", "extra_data"])),
        )

        mock_multi_get.assert_called_once()

        mock_multi_set.assert_called_once()
        set_call_args = mock_multi_set.call_args[0][0]
        assert len(set_call_args) == 4

        assert len(result) == 3
        assert result[0].data == "new_data1"
        assert result[0].extra_data == "new_extra_data1"
        assert result[1].data == "new_data2"
        assert result[1].extra_data == "new_extra_data2"
        assert result[2].data == "data3"
        assert result[2].extra_data == "extra_data3"


@pytest.mark.asyncio
async def test_fetch_async_data_complete_cache_hit(entities: List[SampleEntity], subset_cache: SubsetCache) -> None:
    await subset_cache.cache.multi_set(
        [
            (({"entity_id": "1"}, "data"), "data1"),
            (({"entity_id": "1"}, "extra_data"), "extra_data1"),
            (({"entity_id": "2"}, "data"), "data2"),
            (({"entity_id": "2"}, "extra_data"), "extra_data2"),
            (({"entity_id": "3"}, "data"), "data3"),
            (({"entity_id": "3"}, "extra_data"), "extra_data3"),
        ]
    )

    with patch.object(subset_cache.cache, "multi_get", wraps=subset_cache.cache.multi_get) as mock_multi_get:
        query_fn = AsyncMock()

        result = cast(
            list[SampleEntity],
            await subset_cache.fetch_async_data(query_fn, entities, to_field_reqs(["data", "extra_data"])),
        )

        mock_multi_get.assert_called_once()
        query_fn.assert_not_called()
        assert len(result) == 3
        assert result[0].data == "data1"
        assert result[0].extra_data == "extra_data1"
        assert result[1].data == "data2"
        assert result[1].extra_data == "extra_data2"
        assert result[2].data == "data3"
        assert result[2].extra_data == "extra_data3"


@pytest.mark.asyncio
async def test_fetch_async_data_complete_cache_hit_with_none_marker(
    entities: List[SampleEntity], subset_cache: SubsetCache
) -> None:
    await subset_cache.cache.multi_set(
        [
            (({"entity_id": "1"}, "data"), "data1"),
            (({"entity_id": "1"}, "extra_data"), "extra_data1"),
            (({"entity_id": "2"}, "data"), "data2"),
            (({"entity_id": "2"}, "extra_data"), ValueNotSet),
            (({"entity_id": "3"}, "data"), "data3"),
            (({"entity_id": "3"}, "extra_data"), "extra_data3"),
        ]
    )

    with patch.object(subset_cache.cache, "multi_get", wraps=subset_cache.cache.multi_get) as mock_multi_get:
        query_fn = AsyncMock()

        result = cast(
            list[SampleEntity],
            await subset_cache.fetch_async_data(query_fn, entities, to_field_reqs(["data", "extra_data"])),
        )

        mock_multi_get.assert_called_once()
        query_fn.assert_not_called()
        assert len(result) == 3
        assert result[0].data == "data1"
        assert result[0].extra_data == "extra_data1"
        assert result[1].data == "data2"
        assert result[1].extra_data is None
        assert result[2].data == "data3"
        assert result[2].extra_data == "extra_data3"


@pytest.mark.asyncio
async def test_fetch_async_data_complete_cache_miss(entities: List[SampleEntity], subset_cache: SubsetCache) -> None:
    with (
        patch.object(subset_cache.cache, "multi_get", wraps=subset_cache.cache.multi_get) as mock_multi_get,
        patch.object(subset_cache.cache, "multi_set", wraps=subset_cache.cache.multi_set) as mock_multi_set,
    ):
        query_fn = AsyncMock(return_value=entities)

        result = cast(
            list[SampleEntity],
            await subset_cache.fetch_async_data(query_fn, entities, to_field_reqs(["data", "extra_data"])),
        )

        mock_multi_get.assert_called_once()
        query_fn.assert_called_once()
        mock_multi_set.assert_called_once()
        assert len(result) == 3
        assert result[0].data == "data1"
        assert result[0].extra_data == "extra_data1"
        assert result[1].data == "data2"
        assert result[1].extra_data == "extra_data2"
        assert result[2].data == "data3"
        assert result[2].extra_data == "extra_data3"


@pytest.mark.asyncio
async def test_fetch_async_data_empty_entities_list(subset_cache: SubsetCache) -> None:
    query_fn = AsyncMock()

    result = await subset_cache.fetch_async_data(query_fn, [], to_field_reqs(["data", "extra_data"]))

    query_fn.assert_not_called()
    assert result == []


@pytest.mark.asyncio
async def test_fetch_async_data_empty_fields_list(entities: List[SampleEntity], subset_cache: SubsetCache) -> None:
    query_fn = AsyncMock()

    result = await subset_cache.fetch_async_data(query_fn, entities, [])

    query_fn.assert_not_called()
    assert result == entities


@pytest.mark.asyncio
async def test_fetch_async_data_single_entity_and_field(
    subset_cache: SubsetCache,
) -> None:
    with (
        patch.object(subset_cache.cache, "multi_get", wraps=subset_cache.cache.multi_get) as mock_multi_get,
        patch.object(subset_cache.cache, "multi_set", wraps=subset_cache.cache.multi_set) as mock_multi_set,
    ):
        entity = SampleEntity(id="1", data="data1")
        query_fn = AsyncMock(return_value=[SampleEntity(id="1", data="new_data1")])

        result = cast(
            list[SampleEntity],
            await subset_cache.fetch_async_data(query_fn, [entity], to_field_reqs(["data"])),
        )

        mock_multi_get.assert_called_once_with([({"entity_id": "1"}, "data")])
        mock_multi_set.assert_called_once()
        assert len(result) == 1
        assert result[0].data == "new_data1"


@pytest.mark.asyncio
async def test_put(entities: List[SampleEntity], subset_cache: SubsetCache) -> None:
    with (
        patch.object(subset_cache.cache, "multi_get", wraps=subset_cache.cache.multi_get) as _,
        patch.object(subset_cache.cache, "multi_set", wraps=subset_cache.cache.multi_set) as mock_multi_set,
    ):
        entity_with_some_none_value = SampleEntity(id="4", data="data4", more_data=None, extra_data="extra_data4")
        entities.append(entity_with_some_none_value)
        await subset_cache.put(entities, to_field_reqs(["data", "extra_data", "invalid_field"]))
        mock_multi_set.assert_called_once()

        query_fn = AsyncMock(return_value=[])
        await subset_cache.fetch_async_data(query_fn, entities, to_field_reqs(["data", "extra_data"]))
        assert query_fn.call_count == 0


@pytest.mark.asyncio
async def test_fetch_data_via_computed_field_name(entities: List[SampleEntity], subset_cache: SubsetCache) -> None:
    with (
        patch.object(subset_cache.cache, "multi_get", wraps=subset_cache.cache.multi_get) as mock_multi_get,
        patch.object(subset_cache.cache, "multi_set", wraps=subset_cache.cache.multi_set) as mock_multi_set,
    ):
        extra_data_computation_id = "extra_data_computation"
        extra_data_computation_id_hash = hash(extra_data_computation_id)
        dynamic_fields_computation_ids = {
            "extra_data": extra_data_computation_id,
        }
        SampleEntity.dynamic_fields_computation_ids = dynamic_fields_computation_ids
        query_fn = AsyncMock(
            return_value=[
                SampleEntity(
                    id="1",
                    data="new_data1",
                    extra_data="new_extra_data1",
                ),
                SampleEntity(
                    id="2",
                    data="new_data2",
                    extra_data="new_extra_data2",
                ),
                SampleEntity(
                    id="3",
                    data="data3",
                    extra_data="extra_data3",
                ),
            ]
        )

        first_call_result = cast(
            list[SampleEntity],
            await subset_cache.fetch_async_data(query_fn, entities, to_field_reqs(["data", "extra_data"])),
        )
        await subset_cache.fetch_async_data(query_fn, entities, to_field_reqs(["data"]))
        await subset_cache.fetch_async_data(query_fn, entities, to_field_reqs(["extra_data"]))

        assert mock_multi_get.call_count == 3
        assert mock_multi_get.call_args_list[0].args == (
            [
                ({"entity_id": "1"}, "data"),
                (
                    {
                        "computation_hash": extra_data_computation_id_hash,
                        "entity_id": "1",
                    },
                    "extra_data",
                ),
                ({"entity_id": "2"}, "data"),
                (
                    {
                        "computation_hash": extra_data_computation_id_hash,
                        "entity_id": "2",
                    },
                    "extra_data",
                ),
                ({"entity_id": "3"}, "data"),
                (
                    {
                        "computation_hash": extra_data_computation_id_hash,
                        "entity_id": "3",
                    },
                    "extra_data",
                ),
            ],
        )

        assert mock_multi_set.call_count == 1
        assert mock_multi_set.call_args_list[0].args == (
            [
                (({"entity_id": "1"}, "data"), "new_data1"),
                (({"entity_id": "2"}, "data"), "new_data2"),
                (({"entity_id": "3"}, "data"), "data3"),
                (
                    (
                        {
                            "computation_hash": extra_data_computation_id_hash,
                            "entity_id": "1",
                        },
                        "extra_data",
                    ),
                    "new_extra_data1",
                ),
                (
                    (
                        {
                            "computation_hash": extra_data_computation_id_hash,
                            "entity_id": "2",
                        },
                        "extra_data",
                    ),
                    "new_extra_data2",
                ),
                (
                    (
                        {
                            "computation_hash": extra_data_computation_id_hash,
                            "entity_id": "3",
                        },
                        "extra_data",
                    ),
                    "extra_data3",
                ),
            ],
        )

        query_fn.assert_called_once()
        assert len(first_call_result) == 3
        assert first_call_result[0].data == "new_data1"
        assert first_call_result[0].extra_data == "new_extra_data1"
        assert first_call_result[1].data == "new_data2"
        assert first_call_result[1].extra_data == "new_extra_data2"
        assert first_call_result[2].data == "data3"
        assert first_call_result[2].extra_data == "extra_data3"


@pytest.mark.asyncio
async def test_fetch_data_with_multiple_computation_ids_for_same_field(
    entities: List[SampleEntity], subset_cache: SubsetCache
) -> None:
    with (
        patch.object(subset_cache.cache, "multi_get", wraps=subset_cache.cache.multi_get) as mock_multi_get,
        patch.object(subset_cache.cache, "multi_set", wraps=subset_cache.cache.multi_set) as mock_multi_set,
    ):
        extra_data_computation_id = "extra_data_computation"
        dynamic_fields_computation_ids = {
            "extra_data": extra_data_computation_id,
        }
        SampleEntity.dynamic_fields_computation_ids = dynamic_fields_computation_ids

        query_fn = AsyncMock(
            return_value=[
                SampleEntity(
                    id="1",
                    data="new_data1",
                    extra_data="new_extra_data1",
                ),
                SampleEntity(
                    id="2",
                    data="new_data2",
                    extra_data="new_extra_data2",
                ),
                SampleEntity(
                    id="3",
                    data="data3",
                    extra_data="extra_data3",
                ),
            ]
        )

        await subset_cache.fetch_async_data(query_fn, entities, to_field_reqs(["data", "extra_data"]))

        SampleEntity.dynamic_fields_computation_ids = {
            "extra_data": "another_extra_data_computation",
        }
        await subset_cache.fetch_async_data(query_fn, entities, to_field_reqs(["extra_data"]))
        await subset_cache.fetch_async_data(query_fn, entities, to_field_reqs(["extra_data"]))

        assert mock_multi_get.call_count == 3
        assert mock_multi_set.call_count == 2
        assert query_fn.call_count == 2


@pytest.mark.asyncio
async def test_fetch_async_data_with_doc_loading_error(entities: List[SampleEntity], subset_cache: SubsetCache) -> None:
    """Test that DocLoadingError instances are not cached."""
    with (
        patch.object(subset_cache.cache, "multi_get", wraps=subset_cache.cache.multi_get) as mock_multi_get,
        patch.object(subset_cache.cache, "multi_set", wraps=subset_cache.cache.multi_set) as mock_multi_set,
    ):
        error_entity = DocLoadingError(corpus_id="2", original_exception=Exception("Test error"))

        query_fn = AsyncMock(
            return_value=[
                SampleEntity(id="1", data="new_data1", extra_data="new_extra_data1"),
                SampleEntity(id="2", data=error_entity, extra_data=error_entity),  # This should not be cached
                SampleEntity(id="3", data="new_data3", extra_data="new_extra_data3"),
            ]
        )

        result = await subset_cache.fetch_async_data(query_fn, entities, to_field_reqs(["data", "extra_data"]))
        mock_multi_get.assert_called_once()
        mock_multi_set.assert_called_once()
        set_call_args = mock_multi_set.call_args[0][0]

        # Check that only values from non-error entities are cached
        # We expect 4 values: data and extra_data for entities with id "1" and "3"
        assert len(set_call_args) == 4

        # Verify that none of the cached keys contain the error entity's ID
        for key, _ in set_call_args:
            assert key[0].get("entity_id") != "2"

        # Verify the result contains all entities including the error
        assert len(result) == 3
