import pytest
from ai2i.dcollection import CitationContext, Snippet

from mabool.agents.common.computed_fields.relevant_snippets import _choose_min_combined_span, _fuzzy_find_snippet_text


@pytest.mark.parametrize(
    "text, snippet_to_find, expected",
    [
        # Exact match
        ("This is a test string", "test", [(10, 14)]),
        # Case-insensitive match
        ("This is a Test string", "test", [(10, 14)]),
        # Multiple matches
        ("Test this test string", "test", [(0, 4), (10, 14)]),
        # Match with special characters ignored
        ("This is a t.e.s.t string", "test", [(10, 17)]),
        # Match with spaces ignored
        ("This is a t e s t string", "test", [(10, 17)]),
        # No match
        ("This is a test string", "notfound", []),
        # Match at the beginning
        ("test string", "test", [(0, 4)]),
        # Match at the end
        ("string test", "test", [(7, 11)]),
        # Empty snippet to find
        ("This is a test string", "", []),
        # Empty text
        ("", "test", []),
        # Both text and snippet empty
        ("", "", []),
        # No alphanumeric characters in text
        ("!!!", "test", []),
        # No alphanumeric characters in snippet
        ("This is a test string", "!!!", []),
        # Match something at the end of the text
        ("This, is a test string.", "test.", [(11, 15)]),
        # Match with unicode and diacritics before
        ("TÜBİTAK test", "test", [(8, 12)]),
        # Match with unicode and diacritics after
        ("test TÜBİTAK", "test", [(0, 4)]),
        # Match with diactritics in snippet
        ("This is a test", "tëst", [(10, 14)]),
        # Match with diactritics in text
        ("This is a tëst", "test", [(10, 14)]),
        ("This is a tëst", "tëst", [(10, 14)]),
    ],
)
def test_fuzzy_find_snippet_text(text: str, snippet_to_find: str, expected: list[tuple[int, int]]) -> None:
    assert _fuzzy_find_snippet_text(text, snippet_to_find) == expected


@pytest.mark.parametrize(
    "part_matches_aggregate, expected",
    [
        # Empty case
        ([], []),
        # Single part with single match
        (
            [
                [
                    Snippet(
                        text="test",
                        section_kind="abstract",
                        section_title="abstract",
                        char_start_offset=10,
                        char_end_offset=14,
                    )
                ]
            ],
            [
                Snippet(
                    text="test",
                    section_kind="abstract",
                    section_title="abstract",
                    char_start_offset=10,
                    char_end_offset=14,
                )
            ],
        ),
        # Single part with multiple matches - should pick the first one
        (
            [
                [
                    Snippet(
                        text="test1",
                        section_kind="abstract",
                        section_title="abstract",
                        char_start_offset=10,
                        char_end_offset=15,
                    ),
                    Snippet(
                        text="test2",
                        section_kind="abstract",
                        section_title="abstract",
                        char_start_offset=20,
                        char_end_offset=25,
                    ),
                ]
            ],
            [
                Snippet(
                    text="test1",
                    section_kind="abstract",
                    section_title="abstract",
                    char_start_offset=10,
                    char_end_offset=15,
                )
            ],
        ),
        # Multiple parts with one match each - should find the combination
        (
            [
                [
                    Snippet(
                        text="part1",
                        section_kind="abstract",
                        section_title="abstract",
                        char_start_offset=10,
                        char_end_offset=15,
                    )
                ],
                [
                    Snippet(
                        text="part2",
                        section_kind="abstract",
                        section_title="abstract",
                        char_start_offset=20,
                        char_end_offset=25,
                    )
                ],
            ],
            [
                Snippet(
                    text="part1",
                    section_kind="abstract",
                    section_title="abstract",
                    char_start_offset=10,
                    char_end_offset=15,
                ),
                Snippet(
                    text="part2",
                    section_kind="abstract",
                    section_title="abstract",
                    char_start_offset=20,
                    char_end_offset=25,
                ),
            ],
        ),
        # Multiple parts with multiple matches each - should find min span
        (
            [
                [
                    Snippet(
                        text="part1a",
                        section_kind="abstract",
                        section_title="abstract",
                        char_start_offset=10,
                        char_end_offset=16,
                    ),
                    Snippet(
                        text="part1b",
                        section_kind="abstract",
                        section_title="abstract",
                        char_start_offset=50,
                        char_end_offset=56,
                    ),
                ],
                [
                    Snippet(
                        text="part2a",
                        section_kind="abstract",
                        section_title="abstract",
                        char_start_offset=20,
                        char_end_offset=26,
                    ),
                    Snippet(
                        text="part2b",
                        section_kind="abstract",
                        section_title="abstract",
                        char_start_offset=60,
                        char_end_offset=66,
                    ),
                ],
            ],
            [
                Snippet(
                    text="part1a",
                    section_kind="abstract",
                    section_title="abstract",
                    char_start_offset=10,
                    char_end_offset=16,
                ),
                Snippet(
                    text="part2a",
                    section_kind="abstract",
                    section_title="abstract",
                    char_start_offset=20,
                    char_end_offset=26,
                ),
            ],
        ),
        # Mixed Snippet and CitationContext
        (
            [
                [
                    Snippet(
                        text="part1",
                        section_kind="abstract",
                        section_title="abstract",
                        char_start_offset=10,
                        char_end_offset=15,
                    )
                ],
                [CitationContext(text="part2", source_corpus_id="123")],
            ],
            [
                Snippet(
                    text="part1",
                    section_kind="abstract",
                    section_title="abstract",
                    char_start_offset=10,
                    char_end_offset=15,
                ),
                CitationContext(text="part2", source_corpus_id="123"),
            ],
        ),
        # Some parts missing
        (
            [
                [
                    Snippet(
                        text="part1",
                        section_kind="abstract",
                        section_title="abstract",
                        char_start_offset=10,
                        char_end_offset=15,
                    )
                ],
                [],
                [
                    Snippet(
                        text="part3",
                        section_kind="abstract",
                        section_title="abstract",
                        char_start_offset=20,
                        char_end_offset=25,
                    )
                ],
            ],
            [
                Snippet(
                    text="part1",
                    section_kind="abstract",
                    section_title="abstract",
                    char_start_offset=10,
                    char_end_offset=15,
                ),
                None,
                Snippet(
                    text="part3",
                    section_kind="abstract",
                    section_title="abstract",
                    char_start_offset=20,
                    char_end_offset=25,
                ),
            ],
        ),
    ],
)
def test_choose_min_combined_span(
    part_matches_aggregate: list[list[Snippet | CitationContext]], expected: list[Snippet | CitationContext | None]
) -> None:
    result = _choose_min_combined_span(part_matches_aggregate)
    assert len(result) == len(expected)

    for res, exp in zip(result, expected):
        if res is None:
            assert exp is None
        else:
            # Compare relevant attributes
            if isinstance(exp, Snippet):
                assert isinstance(res, Snippet)
                assert res.text == exp.text
                assert res.section_kind == exp.section_kind
                assert res.section_title == exp.section_title
                assert res.char_start_offset == exp.char_start_offset
                assert res.char_end_offset == exp.char_end_offset
            elif isinstance(exp, CitationContext):
                assert isinstance(res, CitationContext)
                assert res.text == exp.text
                assert res.source_corpus_id == exp.source_corpus_id


def test_combinatorial_explosion() -> None:
    """Test the function's behavior when facing combinatorial explosion."""
    # Create a scenario with many possible combinations (more than 100,000)
    # We'll use 6 parts with 10 matches each (10^6 = 1,000,000 combinations)
    part_matches = []
    for i in range(6):
        part = []
        for j in range(10):
            offset = i * 100 + j * 10
            part.append(
                Snippet(
                    text=f"part{i}_match{j}",
                    section_kind="abstract",
                    section_title="abstract",
                    char_start_offset=offset,
                    char_end_offset=offset + 5,
                )
            )
        part_matches.append(part)

    # The function should handle this by taking the first match from each part
    result = _choose_min_combined_span(part_matches)

    # Verify that it took the first match from each part
    assert len(result) == 6
    for i, res in enumerate(result):
        assert isinstance(res, Snippet)
        assert res.text == f"part{i}_match0"
        assert res.char_start_offset == i * 100
