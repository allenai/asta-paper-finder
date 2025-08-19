from contextlib import nullcontext
from random import shuffle

import pytest

from ai2i.di.interface.errors import CyclicDependecyError
from ai2i.di.interface.models import DependencyDefinition
from ai2i.di.utils import dependency_topological_sort


def test_topological_order_sanity() -> None:
    dep_a = _create_dep("a")
    dep_b = _create_dep("b")
    deps = [dep_a, dep_b]

    ordered_deps = dependency_topological_sort(deps)

    assert len(ordered_deps) == 2
    assert dep_a in ordered_deps
    assert dep_b in ordered_deps


def test_topological_order_basic() -> None:
    dep_a = _create_dep("a")
    dep_b = _create_dep("b", ["a"])
    deps = [dep_a, dep_b]

    ordered_deps = dependency_topological_sort(deps)

    assert len(ordered_deps) == 2
    assert ordered_deps[0].unique_name == "a"
    assert ordered_deps[1].unique_name == "b"

    ordered_deps = dependency_topological_sort(reversed(deps))
    assert len(ordered_deps) == 2
    assert ordered_deps[0].unique_name == "a"
    assert ordered_deps[1].unique_name == "b"


def test_topological_order_has_cyclic_dep() -> None:
    dep_a = _create_dep("a", ["b"])
    dep_b = _create_dep("b", ["a"])
    deps = [dep_a, dep_b]

    with pytest.raises(CyclicDependecyError):
        dependency_topological_sort(deps)


def test_topological_order_long_chain() -> None:
    dep_a = _create_dep("a")
    dep_b = _create_dep("b", ["a"])
    dep_c = _create_dep("c", ["b"])
    dep_d = _create_dep("d", ["c"])
    dep_e = _create_dep("e", ["c", "a", "d"])
    deps = [dep_a, dep_b, dep_c, dep_d, dep_e]

    for _ in range(5):
        shuffle(deps)
        ordered_deps = dependency_topological_sort(deps)

        assert len(ordered_deps) == 5
        assert ordered_deps[0].unique_name == "a"
        assert ordered_deps[1].unique_name == "b"
        assert ordered_deps[2].unique_name == "c"
        assert ordered_deps[3].unique_name == "d"
        assert ordered_deps[4].unique_name == "e"


def test_topological_order_with_ignore() -> None:
    dep_a = _create_dep("a", ["k"])
    dep_b = _create_dep("b", ["a", "k"])
    dep_c = _create_dep("c", ["b"])
    dep_d = _create_dep("d", ["c", "k"])
    dep_e = _create_dep("e", ["c", "a", "d"])
    deps = [dep_a, dep_b, dep_c, dep_d, dep_e]

    for _ in range(5):
        shuffle(deps)
        ordered_deps = dependency_topological_sort(deps, ignore_dep_names=["k"])

        assert len(ordered_deps) == 5
        assert ordered_deps[0].unique_name == "a"
        assert ordered_deps[1].unique_name == "b"
        assert ordered_deps[2].unique_name == "c"
        assert ordered_deps[3].unique_name == "d"
        assert ordered_deps[4].unique_name == "e"


def _create_dep(name: str, deps: list[str] | None = None) -> DependencyDefinition[None]:
    _deps = deps or []
    return DependencyDefinition(
        unique_name=name,
        create_context_manager=lambda _: nullcontext(),
        depends_on=_deps,
    )
