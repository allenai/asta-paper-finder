import ast
from ast import Assign, Attribute, Call, For, Name
from typing import Any, Generator, Type

# TODO: duplicate code, extract to a common place.


class StableSortChecker:
    """Flake8 plugin to enforce stable sorting for pandas methods and prevent set iteration."""

    name = "flake8-stable-sort"
    version = "0.1.0"

    # Pandas methods that require kind='stable'
    PANDAS_SORT_METHODS = {
        "sort_values",
        "sort_index",
    }

    def __init__(self, tree: ast.AST, filename: str):
        self.tree = tree
        self.filename = filename
        self.set_variables: dict[str, bool] = {}

    def _check_pandas_sort(self, node: Call) -> bool:
        """Check if pandas sort method has kind='stable'."""
        return any(
            keyword.arg == "kind" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "stable"
            for keyword in node.keywords
        )

    def _is_set_type(self, node: ast.AST) -> bool:
        """Check if a node represents a set type."""
        # Check for set literal {1, 2, 3}
        if isinstance(node, ast.Set):
            return True

        # Check for set() constructor
        if isinstance(node, Call) and isinstance(node.func, ast.Name) and node.func.id == "set":
            return True

        # Check if it's a variable we know contains a set
        if isinstance(node, Name) and node.id in self.set_variables:
            return True

        return False

    def run(self) -> Generator[tuple[int, int, str, Type[Any]], None, None]:
        # First pass: collect variables assigned sets
        for node in ast.walk(self.tree):
            if isinstance(node, Assign):
                for target in node.targets:
                    if isinstance(target, Name) and self._is_set_type(node.value):
                        self.set_variables[target.id] = True

        # Second pass: check for issues
        for node in ast.walk(self.tree):
            # Check for stable sort requirement
            if isinstance(node, Call) and isinstance(node.func, Attribute):
                if node.func.attr in self.PANDAS_SORT_METHODS:
                    if not self._check_pandas_sort(node):
                        yield (
                            node.lineno,
                            node.col_offset,
                            f'SST001 {node.func.attr}() called without kind="stable" - sorting will be unstable',
                            type(self),
                        )

            # Check for set iteration
            if isinstance(node, For):
                if self._is_set_type(node.iter):
                    yield (
                        node.lineno,
                        node.col_offset,
                        "SST002 Iteration over a set detected - iteration order is non-deterministic",
                        type(self),
                    )
