from __future__ import annotations

from abc import abstractmethod
from typing import Any, Awaitable, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo
from semanticscholar import AsyncSemanticScholar

from ai2i.dcollection.external_api.dense.vespa import VespaRetriever
from ai2i.dcollection.interface.document import DocumentFieldName

type EntityId = str

type FunctionCodeId = str
type KeywordArgs = frozenset[tuple[str, Any]]
type CodeIdOfPartialFunction = tuple[FunctionCodeId, Sequence[Any], KeywordArgs | None]
type ComputationId = FunctionCodeId | CodeIdOfPartialFunction


@runtime_checkable
class SubsetCacheInterface(Protocol):
    def __init__(
        self,
        ttl: int,
        is_enabled: bool = True,
        force_deterministic: bool = False,
    ) -> None: ...

    async def fetch_async_data[DFN: str](
        self,
        query_fn: QueryFnSansContext[DFN],
        entities: Sequence[DynamicallyLoadedEntity[DFN]],
        fields: Sequence[FieldRequirements[DFN]],
    ) -> list[DynamicallyLoadedEntity[DFN]]: ...

    async def put[DFN: str](
        self,
        entities: Sequence[DynamicallyLoadedEntity[DFN]],
        fields: Sequence[FieldRequirements[DFN]],
    ) -> None: ...

    async def clear(self) -> None: ...

    def enabled(self) -> bool: ...


class FieldRequirements[DFN: str]:
    def __init__(self, field: DFN, required_fields: Sequence[DFN] | None, skip_cache: bool = False) -> None:
        self.field = field
        self.required_fields = required_fields or []
        self.skip_cache = skip_cache


class QueryFnSansContext[DFN: str](Protocol):
    def __call__(
        self,
        entities: Sequence[DynamicallyLoadedEntity[DFN]],
        fields: list[DFN],
    ) -> Awaitable[list[DynamicallyLoadedEntity[DFN]]]: ...


class Fuser[DFN: str](Protocol):
    def __call__(
        self, fuse_to: DynamicallyLoadedEntity[DFN], fuse_from: DynamicallyLoadedEntity[DFN], field: DFN
    ) -> None: ...


class Entity(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    @abstractmethod
    def entity_id(self) -> EntityId: ...


class DocumentFieldLoader[D: Entity](Protocol):
    async def __call__(
        self, entities: Sequence[D], fields: Sequence[DocumentFieldName], context: DocumentCollectionContext
    ) -> Sequence[D]: ...


class DynamicField(FieldInfo):
    model_config = ConfigDict(frozen=True)

    fuse: Fuser[DocumentFieldName]
    required_fields: Sequence[DocumentFieldName]
    loading_functions: Sequence[DocumentFieldLoader]
    computation_id: ComputationId | None
    mandatory_loader: bool
    cache: bool
    extra: bool

    def __init__(
        self,
        *args: Any,
        loaders: Sequence[DocumentFieldLoader] | None = None,
        computation_id: ComputationId | None = None,
        fuse: Fuser[DocumentFieldName],
        required_fields: Sequence[DocumentFieldName] | None = None,
        mandatory_loader: bool = False,
        cache: bool = True,
        extra: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fuse = fuse
        self.required_fields = required_fields if required_fields is not None else []
        self.loading_functions = loaders if loaders is not None else []
        self.computation_id = computation_id
        self.mandatory_loader = mandatory_loader
        self.cache = cache
        self.extra = extra

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.required_fields),
                self.computation_id,
                self.mandatory_loader,
                self.cache,
                self.extra,
            )
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DynamicField):
            return NotImplemented
        return bool(
            self.required_fields == other.required_fields
            and self.computation_id == other.computation_id
            and self.mandatory_loader == other.mandatory_loader
            and self.cache == other.cache
            and self.extra == other.extra
        )


class DynamicallyLoadedEntity[DFN: str](Entity):
    dynamic_fields: dict[DFN | str, DynamicField] = Field(default_factory=dict, exclude=True)

    @abstractmethod
    def is_loaded(self, field_name: DFN | str) -> bool: ...

    @abstractmethod
    def clear_loaded_field(self, field_name: DFN | str) -> None: ...

    def get_dynamic_field_computation_id(self, field_name: DFN | str) -> ComputationId | None:
        # TODO: does it need to include fields not in dynamic_fields?
        entity_field = self.dynamic_fields.get(field_name)
        if entity_field and isinstance(entity_field, DynamicField) and entity_field.extra:
            return entity_field.computation_id
        return None


class DocumentCollectionContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    s2_client: AsyncSemanticScholar = Field()
    s2_max_concurrency: int = Field(default=10)
    cache: SubsetCacheInterface = Field()
    vespa_client: VespaRetriever = Field()
    force_deterministic: bool = Field(default=False)
