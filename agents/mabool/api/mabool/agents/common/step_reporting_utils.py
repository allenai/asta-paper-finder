from typing import Literal


def verbalize_list(items: list[str], connector: Literal["and", "or"]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) > 1:
        return f"{', '.join(items[:-1])} {connector} {items[-1]}"
    return "empty set."


def markdown_list(items: list[str], level: int = 1, numbered: bool = False) -> str:
    indent = (level - 1) * 4
    if numbered:
        return "\n".join([f"{'' * indent}{n}. {item}" for n, item in enumerate(items, 1)])
    else:
        return "\n".join([f"{'' * indent}- {item}" for item in items])


def pluralize(base: str, count: int) -> str:
    return base + "s" if count != 1 else base


def counted_noun(base: str, count: int) -> str:
    if count == 1:
        return f"one {base}"
    return f"{count} {base}s"
