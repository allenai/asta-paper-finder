from __future__ import annotations

from dataclasses import dataclass
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Generic,
    Type,
    TypedDict,
    TypeVar,
    TypeVarTuple,
    Unpack,
    overload,
)

from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.base import Runnable

IN = TypeVar("IN", contravariant=True)
OUT = TypeVar("OUT", covariant=True)

NEW_IN = TypeVar("NEW_IN", contravariant=True)
NEW_OUT = TypeVar("NEW_OUT", covariant=True)


IN2 = TypeVar("IN2", contravariant=True)
OUT1 = TypeVar("OUT1", covariant=True)
OUT2 = TypeVar("OUT2", covariant=True)
OUT3 = TypeVar("OUT3", covariant=True)
OUT4 = TypeVar("OUT4", covariant=True)
OUT5 = TypeVar("OUT5", covariant=True)
OUT6 = TypeVar("OUT6", covariant=True)
OUT7 = TypeVar("OUT7", covariant=True)
OUT8 = TypeVar("OUT8", covariant=True)
OUT9 = TypeVar("OUT9", covariant=True)
OUT10 = TypeVar("OUT10", covariant=True)
OUT11 = TypeVar("OUT11", covariant=True)


N = TypeVar("N")

Ts = TypeVarTuple("Ts")

ModelRunnable = Runnable[PromptValue, BaseMessage]
ModelRunnableFactory = Callable[[], ModelRunnable]


LangChainRunnableBuilder = Callable[[ModelRunnableFactory], Runnable[IN, OUT]]


A = TypeVar("A")
B = TypeVar("B")


@dataclass(frozen=True)
class ChainComputation(Generic[IN, OUT]):
    _builder: LangChainRunnableBuilder[IN, OUT]

    @overload
    @staticmethod
    def lift(runnable: Runnable[A, B]) -> ChainComputation[A, B]:
        pass

    @overload
    @staticmethod
    def lift(runnable: Callable[[A], B]) -> ChainComputation[A, B]:
        pass

    @staticmethod
    def lift(runnable: Runnable[A, B] | Callable[[A], B]) -> ChainComputation[A, B]:
        if isinstance(runnable, Runnable):
            return ChainComputation(lambda _: runnable)
        else:
            return ChainComputation(lambda _: RunnableLambda(runnable))

    @staticmethod
    def suspend_runnable(builder: Callable[[], Runnable[A, B]]) -> ChainComputation[A, B]:
        return ChainComputation(lambda _: builder())

    def build_runnable(self, mf: ModelRunnableFactory) -> Runnable[IN, OUT]:
        return self._builder(mf)

    def with_trace_name(self, name: str) -> ChainComputation[IN, OUT]:
        def _internal_with_trace_name(mf: ModelRunnableFactory) -> Runnable[IN, OUT]:
            return self._builder(mf).with_config({"run_name": name})

        return ChainComputation(_internal_with_trace_name)

    # Basic profunctor operations (manipulation of input and manipulation of output)
    def dimap(self, f: Callable[[NEW_IN], IN], g: Callable[[OUT], NEW_OUT]) -> ChainComputation[NEW_IN, NEW_OUT]:
        def _internal_dimap(mf: ModelRunnableFactory) -> Runnable[NEW_IN, NEW_OUT]:
            return RunnableLambda(f) | self._builder(mf) | RunnableLambda(g)

        return ChainComputation(_internal_dimap)

    def map(self, g: Callable[[OUT], NEW_OUT]) -> ChainComputation[IN, NEW_OUT]:
        def _internal_map(mf: ModelRunnableFactory) -> Runnable[IN, NEW_OUT]:
            return self._builder(mf) | RunnableLambda(g)

        return ChainComputation(_internal_map)

    @overload
    def contra_map(self, f: Callable[[NEW_IN], IN], /, input_type: None = None) -> ChainComputation[NEW_IN, OUT]:
        pass

    @overload
    def contra_map(self, f: Callable[[Any], IN], /, input_type: Type[NEW_IN]) -> ChainComputation[NEW_IN, OUT]:
        pass

    def contra_map(self, f: Callable[[Any], Any], /, input_type: Type[Any] | None = None) -> ChainComputation[Any, OUT]:
        def _internal_contra_map(mf: ModelRunnableFactory) -> Runnable[Any, OUT]:
            return RunnableLambda(f) | self._builder(mf)

        return ChainComputation(_internal_contra_map)

    # expose convenient pattern from Runnable
    def passthrough_input(self) -> ChainComputation[IN, tuple[IN, OUT]]:
        def _internal_passthrough_input(mf: ModelRunnableFactory) -> Runnable[IN, tuple[IN, OUT]]:
            return RunnableParallel(
                first=RunnablePassthrough(),
                second=self._builder(mf),
            ) | RunnableLambda(dict2tuple)

        return ChainComputation(_internal_passthrough_input)

    # Cartesian profunctor (allow passthrough of additional information, not digested by the computation)
    # NOTE: a.k.a second in profunctor docs
    def passthrough_as_first(self, pt: Type[N]) -> ChainComputation[tuple[N, IN], tuple[N, OUT]]:
        def _internal_passthrough_first(mf: ModelRunnableFactory) -> Runnable[tuple[N, IN], tuple[N, OUT]]:
            return RunnableParallel(
                first=RunnablePassthrough() | RunnableLambda(itemgetter(0)),
                second=RunnableLambda(itemgetter(1)) | self._builder(mf),
            ) | RunnableLambda(dict2tuple)

        return ChainComputation(_internal_passthrough_first)

    # NOTE: a.k.a first in profunctor docs
    def passthrough_as_second(self, pt: Type[N]) -> ChainComputation[tuple[IN, N], tuple[OUT, N]]:
        def _internal_passthrough_second(mf: ModelRunnableFactory) -> Runnable[tuple[IN, N], tuple[OUT, N]]:
            return RunnableParallel(
                first=RunnableLambda(itemgetter(0)) | self._builder(mf),
                second=RunnablePassthrough() | RunnableLambda(itemgetter(1)),
            ) | RunnableLambda(dict2tuple)

        return ChainComputation(_internal_passthrough_second)

    # Pipe prompts
    def pipe_to(self, p: ChainComputation[OUT, NEW_OUT]) -> ChainComputation[IN, NEW_OUT]:
        def _internal_and_then(mf: ModelRunnableFactory) -> Runnable[IN, NEW_OUT]:
            return self._builder(mf) | p._builder(mf)

        return ChainComputation(_internal_and_then)

    # sequencing operator
    def __or__(self, p: ChainComputation[OUT, NEW_OUT]) -> ChainComputation[IN, NEW_OUT]:
        return self.pipe_to(p)

    # Parallel execution
    def in_parllel_with(self, p: ChainComputation[IN2, OUT2]) -> ChainComputation[tuple[IN, IN2], tuple[OUT, OUT2]]:
        def _internal_in_parallel_with(mf: ModelRunnableFactory) -> Runnable[tuple[IN, IN2], tuple[OUT, OUT2]]:
            return RunnableParallel(
                first=RunnableLambda(itemgetter(0)) | self._builder(mf),
                second=RunnableLambda(itemgetter(1)) | p._builder(mf),
            ) | RunnableLambda(dict2tuple)

        return ChainComputation(_internal_in_parallel_with)

    def product(self, p: ChainComputation[IN, OUT2]) -> ChainComputation[IN, tuple[OUT, OUT2]]:
        def _internal_product(mf: ModelRunnableFactory) -> Runnable[IN, tuple[OUT, OUT2]]:
            return RunnableParallel(
                first=self._builder(mf),
                second=p._builder(mf),
            ) | RunnableLambda(dict2tuple)

        return ChainComputation(_internal_product)

    # product operator
    def __pow__(self, p: ChainComputation[IN, OUT2]) -> ChainComputation[IN, tuple[OUT, OUT2]]:
        return self.product(p)

    # Applicative map_n (2 arguments variant)
    @staticmethod
    @overload
    def map_n(
        f: Callable[[OUT, OUT2], NEW_OUT],
        p1: ChainComputation[IN, OUT],
        p2: ChainComputation[IN, OUT2],
        /,
    ) -> ChainComputation[IN, NEW_OUT]:
        pass

    # Applicative map_n (3 arguments variant)
    @overload
    @staticmethod
    def map_n(
        f: Callable[[OUT1, OUT2, OUT3], NEW_OUT],
        p1: ChainComputation[IN2, OUT1],
        p2: ChainComputation[IN2, OUT2],
        p3: ChainComputation[IN2, OUT3],
        /,
    ) -> ChainComputation[IN2, NEW_OUT]:
        pass

    # Applicative map_n (4 arguments variant)
    @overload
    @staticmethod
    def map_n(
        f: Callable[[OUT1, OUT2, OUT3, OUT4], NEW_OUT],
        p1: ChainComputation[IN2, OUT1],
        p2: ChainComputation[IN2, OUT2],
        p3: ChainComputation[IN2, OUT3],
        p4: ChainComputation[IN2, OUT4],
        /,
    ) -> ChainComputation[IN2, NEW_OUT]:
        pass

    # Applicative map_n (5 arguments variant)
    @overload
    @staticmethod
    def map_n(
        f: Callable[[OUT1, OUT2, OUT3, OUT4, OUT5], NEW_OUT],
        p1: ChainComputation[IN2, OUT1],
        p2: ChainComputation[IN2, OUT2],
        p3: ChainComputation[IN2, OUT3],
        p4: ChainComputation[IN2, OUT4],
        p5: ChainComputation[IN2, OUT5],
        /,
    ) -> ChainComputation[IN2, NEW_OUT]:
        pass

    # Applicative map_n (6 arguments variant)
    @overload
    @staticmethod
    def map_n(
        f: Callable[[OUT1, OUT2, OUT3, OUT4, OUT5, OUT6], NEW_OUT],
        p1: ChainComputation[IN2, OUT1],
        p2: ChainComputation[IN2, OUT2],
        p3: ChainComputation[IN2, OUT3],
        p4: ChainComputation[IN2, OUT4],
        p5: ChainComputation[IN2, OUT5],
        p6: ChainComputation[IN2, OUT6],
        /,
    ) -> ChainComputation[IN2, NEW_OUT]:
        pass

    # Applicative map_n (7 arguments variant)
    @overload
    @staticmethod
    def map_n(
        f: Callable[[OUT1, OUT2, OUT3, OUT4, OUT5, OUT6, OUT7], NEW_OUT],
        p1: ChainComputation[IN2, OUT1],
        p2: ChainComputation[IN2, OUT2],
        p3: ChainComputation[IN2, OUT3],
        p4: ChainComputation[IN2, OUT4],
        p5: ChainComputation[IN2, OUT5],
        p6: ChainComputation[IN2, OUT6],
        p7: ChainComputation[IN2, OUT7],
        /,
    ) -> ChainComputation[IN2, NEW_OUT]:
        pass

    # Applicative map_n (8 arguments variant)
    @overload
    @staticmethod
    def map_n(
        f: Callable[[OUT1, OUT2, OUT3, OUT4, OUT5, OUT6, OUT7, OUT8], NEW_OUT],
        p1: ChainComputation[IN2, OUT1],
        p2: ChainComputation[IN2, OUT2],
        p3: ChainComputation[IN2, OUT3],
        p4: ChainComputation[IN2, OUT4],
        p5: ChainComputation[IN2, OUT5],
        p6: ChainComputation[IN2, OUT6],
        p7: ChainComputation[IN2, OUT7],
        p8: ChainComputation[IN2, OUT8],
        /,
    ) -> ChainComputation[IN2, NEW_OUT]:
        pass

    # Applicative map_n (9 arguments variant)
    @overload
    @staticmethod
    def map_n(
        f: Callable[[OUT1, OUT2, OUT3, OUT4, OUT5, OUT6, OUT7, OUT8, OUT9], NEW_OUT],
        p1: ChainComputation[IN2, OUT1],
        p2: ChainComputation[IN2, OUT2],
        p3: ChainComputation[IN2, OUT3],
        p4: ChainComputation[IN2, OUT4],
        p5: ChainComputation[IN2, OUT5],
        p6: ChainComputation[IN2, OUT6],
        p7: ChainComputation[IN2, OUT7],
        p8: ChainComputation[IN2, OUT8],
        p9: ChainComputation[IN2, OUT9],
        /,
    ) -> ChainComputation[IN2, NEW_OUT]:
        pass

    # Applicative map_n (10 arguments variant)
    @overload
    @staticmethod
    def map_n(
        f: Callable[[OUT1, OUT2, OUT3, OUT4, OUT5, OUT6, OUT7, OUT8, OUT9, OUT10], NEW_OUT],
        p1: ChainComputation[IN2, OUT1],
        p2: ChainComputation[IN2, OUT2],
        p3: ChainComputation[IN2, OUT3],
        p4: ChainComputation[IN2, OUT4],
        p5: ChainComputation[IN2, OUT5],
        p6: ChainComputation[IN2, OUT6],
        p7: ChainComputation[IN2, OUT7],
        p8: ChainComputation[IN2, OUT8],
        p9: ChainComputation[IN2, OUT9],
        p10: ChainComputation[IN2, OUT10],
        /,
    ) -> ChainComputation[IN2, NEW_OUT]:
        pass

    # Applicative map_n (11 arguments variant)
    @overload
    @staticmethod
    def map_n(
        f: Callable[[OUT1, OUT2, OUT3, OUT4, OUT5, OUT6, OUT7, OUT8, OUT9, OUT10, OUT11], NEW_OUT],
        p1: ChainComputation[IN2, OUT1],
        p2: ChainComputation[IN2, OUT2],
        p3: ChainComputation[IN2, OUT3],
        p4: ChainComputation[IN2, OUT4],
        p5: ChainComputation[IN2, OUT5],
        p6: ChainComputation[IN2, OUT6],
        p7: ChainComputation[IN2, OUT7],
        p8: ChainComputation[IN2, OUT8],
        p9: ChainComputation[IN2, OUT9],
        p10: ChainComputation[IN2, OUT10],
        p11: ChainComputation[IN2, OUT11],
        /,
    ) -> ChainComputation[IN2, NEW_OUT]:
        pass

    # Applicative map_n implementation
    @staticmethod
    def map_n(  # type: ignore
        f: Callable[[Unpack[Ts]], Any], *ps: ChainComputation[IN2, Any]
    ) -> ChainComputation[IN2, Any]:
        def _map_n_f_application(input: dict[str, Any]) -> Any:
            # inputs are a dictionary from index (as string) to result, we first need to convert the results
            # into an ordered list based on the indexes
            ordered_inputs = tuple(x[1] for x in sorted([(int(si), n) for si, n in input.items()], key=lambda x: x[0]))
            return f(*ordered_inputs)  # type: ignore

        def _internal_product(mf: ModelRunnableFactory) -> Runnable[IN2, Any]:
            ps_dict = {str(i): p for i, p in enumerate([p.build_runnable(mf) for p in ps])}
            return RunnableParallel(ps_dict) | RunnableLambda(_map_n_f_application).with_config(
                {"run_name": "map_n: apply f"}
            )

        return ChainComputation(_internal_product)


# -------------- #
# Util functions #
# -------------- #

# NOTE: langchain doesn't support tuples very well so we internally convert from dicts
#       that represent a tuple when using ParallelRunnable


T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


class DictTuple2(Generic[T1, T2], TypedDict):
    first: T1
    second: T2


def dict2tuple(d: DictTuple2[T1, T2]) -> tuple[T1, T2]:
    return d["first"], d["second"]
