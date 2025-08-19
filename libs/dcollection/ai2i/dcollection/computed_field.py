from __future__ import annotations

from types import CodeType
from typing import Any, Awaitable, Callable, Sequence

from pydantic.fields import Field

from ai2i.dcollection.interface.collection import (
    BaseComputedField,
    DocLoadingError,
    Document,
)
from ai2i.dcollection.interface.document import DocumentFieldName


class Typed[T, U]:
    f: Callable[[T], U]

    def __init__(self, f: Callable[[T], U]) -> None:
        self.f = f

    def __call__(self, *args: Any, **kwargs: U) -> Any:
        return self.f(*args, **kwargs)

    @property
    def __code__(self) -> CodeType:
        return self.f.__code__


class DocComputedField[V](BaseComputedField[DocumentFieldName, V]):
    pass


class AssignedField[V](DocComputedField[V]):
    """
    A field that is assigns an externally computed value when the document is created.
    It doesn't have a computation function, or a list of required fields, instead is received a list of values
    to be assigned on the documents currently in the collection.
    To be used when the value cannot be recomputed based on other fields or for additional documents in the collection.
    """

    assigned_values: list[V]

    @property
    def use_cache(self) -> bool:
        return False

    @property
    def computation(self) -> Callable:
        return lambda: (_ for _ in ()).throw(NotImplementedError("AssignedField does not have a computation function"))

    def values_to_docs(self, docs: Sequence[Document]) -> None:
        for doc in docs:
            doc[self.field_name] = self.assigned_values.pop(0)


class ComputedField[V](DocComputedField[V]):
    """
    A simple computed field that is computed on a single document.
    To be used when the value can be recomputed based on other fields or for additional documents in the collection,
    and the operation doesn't involve IO/async operations, so there's no need to batch the computation.
    """

    @property
    def use_cache(self) -> bool:
        return self.cache

    cache: bool = Field(default=True)

    @property
    def computation(self) -> Callable[[Document], V]:
        return self.computation_func

    computation_func: Callable[[Document], V] = Field(exclude=True)


class BatchComputedField[V](DocComputedField[V]):
    """
    A computed field that is computed on a batch of documents.
    To be used when the value can be recomputed based on other fields or for additional documents in the collection,
    and the the operation involves IO/async operations, so we can batch the computation, to limit the number of calls.
    """

    @property
    def use_cache(self) -> bool:
        return self.cache

    cache: bool = Field(default=True)

    @property
    def computation(
        self,
    ) -> Callable[[Sequence[Document]], Awaitable[Sequence[V | DocLoadingError]]]:
        return self.computation_func

    computation_func: Callable[[Sequence[Document]], Awaitable[Sequence[V | DocLoadingError]]] = Field(exclude=True)


class AggTransformComputedField[V](BatchComputedField[V]):
    """
    A computed field that is computed on a batch of documents, but whose value potentially depends on the other
    documents in the collection, so it can't be computed on a single document,
    and whose value therefore should not be cached.
    """

    @property
    def use_cache(self) -> bool:
        return self.cache

    cache: bool = Field(default=False)
