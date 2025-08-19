"""
Test validation functions of specifications in the Mabool data model.
"""

import pytest

from mabool.data_model import specifications as specs


def test_no_contradiction() -> None:
    specs.PaperSpec(
        citing=specs.PaperSpec(
            authors=specs.AuthorSpec(name="Alice"),
        ),
        exclude=specs.PaperSpec(
            authors=specs.AuthorSpec(name="Bob"),
        ),
    )


def test_self_contradiction() -> None:
    with pytest.raises(ValueError, match="Field 'authors' cannot be both set and excluded with the same value"):
        specs.PaperSpec(
            authors=specs.AuthorSpec(name="Alice"),
            exclude=specs.PaperSpec(
                authors=specs.AuthorSpec(name="Alice"),
            ),
        )


def test_not_trivial_metadata_only_if_content() -> None:
    spec = specs.PaperSpec(name="Test Paper", content="This is a test paper.", authors=specs.AuthorSpec(name="Alice"))
    assert not spec.is_non_trivial_metadata_only()


def test_not_trivial_metadata_only_if_name() -> None:
    spec = specs.PaperSpec(name="Test Paper", authors=specs.AuthorSpec(name="Alice"))
    assert not spec.is_non_trivial_metadata_only()


def test_is_non_trivial_metadata_only() -> None:
    spec = specs.PaperSpec(
        authors=specs.AuthorSpec(name="Alice"),
        years=specs.Years(start=2020, end=2021),
    )
    assert spec.is_non_trivial_metadata_only()
