from dataclasses import replace
from typing import Any, Iterable, Sequence

from ai2i.di.interface.errors import CyclicDependecyError
from ai2i.di.interface.models import DependencyDefinition


def dependency_topological_sort(
    defs: Iterable[DependencyDefinition[Any]], ignore_dep_names: list[str] | None = None
) -> Sequence[DependencyDefinition[Any]]:
    """
    Topological sort of dependency definitions based on their defined dependencies
    This will dictate the initialization order of the definitions

    `ignore_dep_names`: dependency names to ignore during sort (some dependencies are provided prior to this sort)
    """
    if ignore_dep_names is None:
        ignore_dep_names = []

    unsorted_defs: list[DependencyDefinition[Any]] = list(defs)
    sorted_defs: list[DependencyDefinition[Any]] = []

    # remove any dependencies that need to be ignored
    unsorted_defs = [
        replace(d, depends_on=[i for i in d.depends_on if i not in ignore_dep_names]) for d in unsorted_defs
    ]

    while len(unsorted_defs) > 0:
        try:
            next_def = next(filter(lambda d: len(d.depends_on) == 0, unsorted_defs))
            unsorted_defs.remove(next_def)
            sorted_defs.append(next_def)

            # remove this dependency from all unsoretd defs
            unsorted_defs = [d.without_dep(next_def.unique_name) for d in unsorted_defs]
        except StopIteration:
            cant_create_deps = ", ".join(d.unique_name for d in unsorted_defs)
            raise CyclicDependecyError(
                f"Unable to create dependencies, because of a cyclic depedndecy: [{cant_create_deps}]"
            )

    return sorted_defs
