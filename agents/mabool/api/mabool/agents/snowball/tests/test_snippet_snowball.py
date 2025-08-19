from ai2i.dcollection import (
    Offset,
    RefMention,
    SentenceOffsets,
    Snippet,
)

from mabool.agents.snowball.snippet_snowball import (
    split_snippets_to_sentence_snippets,
    split_text_to_sentence_offsets,
)


def test_split_text_to_sentence_offsets() -> None:
    snippet = Snippet(
        text="This is a sentence. This is another sentence! And yet another one?",
        char_start_offset=0,
    )
    expected_offsets = [
        SentenceOffsets(
            within_snippet_offset=Offset(start=0, end=20),
            global_offset=Offset(start=0, end=20),
        ),
        SentenceOffsets(
            within_snippet_offset=Offset(start=20, end=46),
            global_offset=Offset(start=20, end=46),
        ),
        SentenceOffsets(
            within_snippet_offset=Offset(start=46, end=66),
            global_offset=Offset(start=46, end=66),
        ),
    ]
    result = split_text_to_sentence_offsets(snippet)
    assert result == expected_offsets


def test_split_text_to_sentence_offsets_with_special_cases() -> None:
    snippet = Snippet(
        text="This is a sentence by Doe et. al. that keeps going till the end of the text. This is another sentence!",
        char_start_offset=0,
    )
    expected_offsets = [
        SentenceOffsets(
            within_snippet_offset=Offset(start=0, end=77),
            global_offset=Offset(start=0, end=77),
        ),
        SentenceOffsets(
            within_snippet_offset=Offset(start=77, end=102),
            global_offset=Offset(start=77, end=102),
        ),
    ]
    result = split_text_to_sentence_offsets(snippet)
    assert result == expected_offsets


def test_split_text_to_sentence_offsets_with_no_char_start_offset() -> None:
    snippet = Snippet(
        text="This is a sentence. This is another sentence!",
        char_start_offset=None,
    )
    expected_offsets = [
        SentenceOffsets(
            within_snippet_offset=Offset(start=0, end=20),
            global_offset=Offset(start=0, end=20),
        ),
        SentenceOffsets(
            within_snippet_offset=Offset(start=20, end=45),
            global_offset=Offset(start=20, end=45),
        ),
    ]
    result = split_text_to_sentence_offsets(snippet)
    assert result == expected_offsets


def test_split_text_to_sentence_offsets_with_empty_text() -> None:
    snippet = Snippet(
        text="",
        char_start_offset=0,
    )
    expected_offsets: list[SentenceOffsets] = []
    result = split_text_to_sentence_offsets(snippet)
    assert result == expected_offsets


def test_split_snippets_to_sentence_snippets_no_ref_mentions() -> None:
    snippet = Snippet(
        text="This is a sentence. This is another sentence!",
        char_start_offset=0,
        ref_mentions=[],
    )
    expected_snippets = [
        Snippet(
            text="This is a sentence. ",
            similarity_scores=None,
            char_start_offset=0,
            char_end_offset=20,
            ref_mentions=None,
        ),
        Snippet(
            text="This is another sentence!",
            similarity_scores=None,
            char_start_offset=20,
            char_end_offset=45,
            ref_mentions=None,
        ),
    ]
    result = split_snippets_to_sentence_snippets(snippet)
    assert result == expected_snippets


def test_split_snippets_to_sentence_snippets_with_ref_mention() -> None:
    snippet = Snippet(
        text="This is a sentence by Doe et. al. that keeps going till the end of the text. This is another sentence!",
        char_start_offset=0,
        ref_mentions=[
            RefMention(matched_paper_corpus_id="123", within_snippet_offset_start=10, within_snippet_offset_end=18)
        ],
    )
    expected_snippets = [
        Snippet(
            text="This is a sentence by Doe et. al. that keeps going till the end of the text. ",
            similarity_scores=None,
            char_start_offset=0,
            char_end_offset=77,
            ref_mentions=[
                RefMention(matched_paper_corpus_id="123", within_snippet_offset_start=10, within_snippet_offset_end=18)
            ],
        ),
        Snippet(
            text="This is another sentence!",
            similarity_scores=None,
            char_start_offset=77,
            char_end_offset=102,
            ref_mentions=[],
        ),
    ]
    result = split_snippets_to_sentence_snippets(snippet)
    assert result == expected_snippets


def test_split_snippets_to_sentence_snippets_with_no_char_start_offset() -> None:
    snippet = Snippet(
        text="This is a sentence. This is another sentence!",
        char_start_offset=None,
        ref_mentions=[
            RefMention(matched_paper_corpus_id="123", within_snippet_offset_start=10, within_snippet_offset_end=18)
        ],
    )
    expected_snippets = [snippet]
    result = split_snippets_to_sentence_snippets(snippet)
    assert result == expected_snippets


def test_split_snippets_to_sentence_snippets_with_empty_text() -> None:
    snippet = Snippet(
        text="",
        char_start_offset=0,
    )
    expected_snippets: list[Snippet] = [snippet]
    result = split_snippets_to_sentence_snippets(snippet)
    assert result == expected_snippets
