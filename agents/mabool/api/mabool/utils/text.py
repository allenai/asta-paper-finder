LIST_SEPARATOR = ", "
AND_CONNECTOR = " and "
COMMA_CONNECTOR = ", "
JOIN_SUFFIX = "."
HRULE = "---"


def join_paragraphs(*args: str) -> str:
    return "\n\n".join(filter(lambda x: len(x) != 0, args))
