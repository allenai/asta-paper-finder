from __future__ import annotations

import logging
from typing import Sequence

from ai2i.dcollection.data_access_context import DocumentCollectionContext
from ai2i.dcollection.interface.collection import Document
from ai2i.dcollection.interface.document import DocumentFieldName

logger = logging.getLogger(__name__)


async def load_markdown(
    entities: Sequence[Document],
    fields: Sequence[DocumentFieldName],
    context: DocumentCollectionContext,
) -> Sequence[Document]:
    for doc in entities:
        markdown = f"# Title: {doc.title}\n"
        if doc.authors:
            markdown += f"## Authors: {', '.join([author.name for author in doc.authors[:3]])}"
            if len(doc.authors) > 3:
                markdown += " et al."
            markdown += "\n"
        if doc.year:
            markdown += f"## Year: {doc.year}\n"
        if doc.abstract:
            markdown += f"## Abstract\n{doc.abstract[:1500]}\n"
        if doc.snippets:
            current_section = None
            for text in doc.snippets:
                if text.section_title and (
                    (text.section_title.lower() == "abstract" and doc.abstract)
                    or (text.section_title.lower() == "title" and doc.title)
                ):
                    continue
                if text.section_title != current_section:
                    current_section = text.section_title
                    markdown += f"\n## Section: {text.section_title}\n...\n"
                markdown += f"{text.text}\n...\n"
        if doc.citation_contexts:
            markdown += (
                "\n# Here are some mentions of the paper taken from other academic papers. "
                "These may add context which is not available in the provided document text "
                "and help make a better judgement (when available the citation is marked "
                "with `<<<...>>>` to help identify it):\n"
            )
            for citation_context in doc.citation_contexts[
                :20
            ]:  # TODO: do we need to limit this? do we want to limit somewhere else?
                markdown += f"- {citation_context.mark_within_snippet_offset()}\n"
        doc.markdown = markdown

    return list(entities)
