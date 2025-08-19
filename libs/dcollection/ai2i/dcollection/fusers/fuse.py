from ai2i.dcollection.data_access_context import DynamicallyLoadedEntity
from ai2i.dcollection.interface.document import BoundingBox, OriginQuery, Snippet


def fuse_origin_query[DFN: str](
    fuse_to: DynamicallyLoadedEntity[DFN], fuse_from: DynamicallyLoadedEntity[DFN], field: str
) -> None:
    fuse_to_origin_queries = getattr(fuse_to, field) if fuse_to.is_loaded(field) else []
    fuse_from_origin_queries = getattr(fuse_from, field) if fuse_from.is_loaded(field) else []

    if fuse_to_origin_queries or fuse_from_origin_queries:
        origin_query_map: dict[OriginQuery, OriginQuery] = {}

        for oq in fuse_to_origin_queries + fuse_from_origin_queries:
            if oq in origin_query_map:
                existing_oq = origin_query_map[oq]
                # Merge ranks, ensuring uniqueness and sorted order
                merged_ranks = sorted(set((existing_oq.ranks or []) + (oq.ranks or [])))
                existing_oq.ranks = merged_ranks
                origin_query_map[oq] = existing_oq
            else:
                origin_query_map[oq] = oq

        # Set the fused result back to fuse_to
        setattr(fuse_to, field, list(origin_query_map.values()))


def fuse_snippet[DFN: str](
    fuse_to: DynamicallyLoadedEntity[DFN], fuse_from: DynamicallyLoadedEntity[DFN], field: str
) -> None:
    fuse_to_snippets = getattr(fuse_to, field) if fuse_to.is_loaded(field) else []
    fuse_from_snippets = getattr(fuse_from, field) if fuse_from.is_loaded(field) else []
    if fuse_to_snippets or fuse_from_snippets:
        fused_sorted_snippets = sorted(_merge_duplicate_snippets(fuse_to_snippets + fuse_from_snippets))
        fused_cleaned_snippets = clean_overlaps(fused_sorted_snippets)
        setattr(fuse_to, field, fused_cleaned_snippets)


def _merge_duplicate_snippets(snippets: list[Snippet]) -> list[Snippet]:
    merged_snippets: dict[Snippet, Snippet] = {}
    for snippet in snippets:
        if snippet in merged_snippets:
            merged_snippets[snippet].similarity_scores = sorted(
                set((merged_snippets[snippet].similarity_scores or []) + (snippet.similarity_scores or [])),
                reverse=True,
            )
        else:
            merged_snippets[snippet] = snippet
    return list(merged_snippets.values())


def clean_overlaps(sorted_snippets: list[Snippet]) -> list[Snippet]:
    no_overlapping_snippets = []
    for i in range(len(sorted_snippets) - 1):
        snippet = sorted_snippets[i]
        next_snippet = sorted_snippets[i + 1]
        if (
            snippet.char_end_offset is not None
            and snippet.char_start_offset is not None
            and next_snippet.char_start_offset is not None
            and (snippet.section_kind == next_snippet.section_kind)
            and (snippet.char_end_offset > next_snippet.char_start_offset)
        ):
            bbs: list[BoundingBox] | None = snippet.bounding_boxes
            if snippet.bounding_boxes and next_snippet.bounding_boxes:
                next_bbs = [bb.model_dump_json() for bb in next_snippet.bounding_boxes]
                bbs = [bb for bb in snippet.bounding_boxes if bb.model_dump_json() not in next_bbs]
            no_overlapping_snippets.append(
                Snippet(
                    text=snippet.text[: next_snippet.char_start_offset - snippet.char_start_offset],
                    section_title=snippet.section_title,
                    section_kind=snippet.section_kind,
                    ref_mentions=snippet.ref_mentions,
                    char_start_offset=snippet.char_start_offset,
                    char_end_offset=next_snippet.char_start_offset,
                    similarity_scores=snippet.similarity_scores,
                    bounding_boxes=bbs,
                )
            )
        else:
            no_overlapping_snippets.append(snippet)

    # add the last one to the list
    no_overlapping_snippets.append(sorted_snippets[-1])

    return no_overlapping_snippets
