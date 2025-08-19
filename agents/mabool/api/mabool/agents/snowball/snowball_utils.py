from typing import Literal

from ai2i.dcollection import DocumentCollection, OriginQuery


def add_snowball_origins(
    promoted_docs: DocumentCollection, variant: Literal["forward", "backward", "snippets"], search_iteration: int
) -> DocumentCollection:
    promoted_docs = promoted_docs.map_enumerate(
        lambda i, doc: doc.clone_with(
            {
                "origins": (doc.origins or [])
                + [
                    OriginQuery(
                        query_type="snowball",
                        query="",
                        variant=variant,
                        iteration=search_iteration,
                        ranks=[i + 1],
                    )
                ],
            },
        )
    )
    return promoted_docs
