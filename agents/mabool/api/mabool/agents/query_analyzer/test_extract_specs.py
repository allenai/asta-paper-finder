# portfolio - ignore

from dataclasses import dataclass

import pytest

from mabool.data_model import specifications as specs

from .query_analyzer import extract_specifications


@dataclass
class Test:
    why: str
    query: str
    spec: specs.PaperSpec

    def id(self) -> str:
        return self.why


testdata: list[Test] = [
    Test(
        "venue and year separately",
        "ASE 2023 paper",
        specs.PaperSpec(years=specs.Years(start=2023, end=2023), venue="ASE"),
    ),
    Test(
        "cited by any other paper",
        "Papers by Mayer Godberg cited by any other paper",
        specs.PaperSpec(authors=specs.AuthorSpec(name="Mayer Godberg"), min_citations=1),
    ),
    Test(
        "publication type JournalArticle",
        "Journal papers by Mayer Godberg",
        specs.PaperSpec(authors=specs.AuthorSpec(name="Mayer Godberg"), publication_type="JournalArticle"),
    ),
    Test(
        "Multiple authors - and",
        "Papers by Mayer Godberg and Yossi Matias",
        specs.PaperSpec(
            authors=specs.Set(
                items=[specs.AuthorSpec(name="Mayer Godberg"), specs.AuthorSpec(name="Yossi Matias")], op="and"
            )
        ),
    ),
    Test(
        "Field of study - given",
        "Papers by David Harel about biology",
        specs.PaperSpec(authors=specs.AuthorSpec(name="David Harel"), field_of_study="biology"),
    ),
    Test(
        "Field of study - not in list",
        "Papers by David Harel about quantum computing",
        specs.PaperSpec(authors=specs.AuthorSpec(name="David Harel"), content="quantum computing"),
    ),
    Test(
        "Min. authors of a paper",
        "NAACL papers by at least 2 of the authors of the 'BERT' paper",
        specs.PaperSpec(
            venue="NAACL",
            authors=specs.AuthorSpec(
                papers=specs.PaperSet(
                    op="any_author_of",
                    items=[
                        specs.PaperSpec(
                            name="BERT",
                            full_name="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                        )
                    ],
                ),
                min_authors=2,
            ),
        ),
    ),
    # NOTE: The following test is commented out because it often fails validation
    # Test(
    #     "Author and any coauthor",
    #     "Papers by Mayer Godberg and any additional coauthor",
    #     specs.PaperSpec(authors=specs.AuthorSpec(name="Mayer Godberg"), min_total_authors=2),
    # ),
    Test(
        "N authors of a paper and M coauthors",
        "Papers by at least 2 of the authors of the 'BERT' paper and at least 3 coauthors",
        specs.PaperSpec(
            authors=specs.AuthorSpec(
                papers=specs.PaperSet(
                    op="any_author_of",
                    items=[
                        specs.PaperSpec(
                            name="BERT",
                            full_name="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                        )
                    ],
                ),
                min_authors=2,
            ),
            min_total_authors=5,
        ),
    ),
    Test(
        "Venue group",
        "IEEE papers about biology from 2010 to 2015",
        specs.PaperSpec(venue_group="IEEE", content="biology", years=specs.Years(start=2010, end=2015)),
    ),
    Test(
        "Negation - not by author",
        "Papers by Andrej Karpathy but not by Furu Wei",
        specs.PaperSpec(
            authors=specs.AuthorSpec(name="Andrej Karpathy"),
            exclude=specs.PaperSpec(authors=specs.AuthorSpec(name="Furu Wei")),
        ),
    ),
    Test(
        "Negation - not in venue",
        "Papers by Andrej Karpathy but not in NeurIPS",
        specs.PaperSpec(authors=specs.AuthorSpec(name="Andrej Karpathy"), exclude=specs.PaperSpec(venue="NeurIPS")),
    ),
    Test(
        "Negation - do not cite",
        "Papers by Andrej Karpathy that do not cite the 'BERT' paper",
        specs.PaperSpec(
            authors=specs.AuthorSpec(name="Andrej Karpathy"),
            exclude=specs.PaperSpec(
                citing=specs.PaperSpec(
                    name="BERT",
                    full_name="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                )
            ),
        ),
    ),
]


@pytest.mark.parametrize("test", testdata, ids=Test.id)
async def test_extract_specs(test: Test) -> None:
    result = await extract_specifications(test.query)
    match test.spec:
        case specs.PaperSpec():
            expected = specs.Specifications(union=[test.spec])
        case specs.Specifications():
            expected = test.spec
    assert expected == result


@pytest.mark.xfail(reason="Understanding self-citations constraints is very hard")
async def test_extract_specs_self_citation() -> None:
    query = "Papers by Andrej Karpathy that cite Furu Wei but are not self-citations of Furu Wei"
    result = await extract_specifications(query)
    match result:
        case specs.Specifications(
            union=[
                specs.PaperSpec(
                    authors=specs.AuthorSpec(name="Andrej Karpathy"),
                    citing=specs.PaperSpec(
                        authors=specs.AuthorSpec(name="Furu Wei"),
                    ),
                    exclude=specs.PaperSpec(authors=specs.AuthorSpec(name="Furu Wei")),
                )
            ]
        ):
            assert True
        case _:
            raise ValueError(f"Unexpected result: {result}")


class TestContent:
    async def test_extract_content_only(self) -> None:
        query = "Papers about asynchronous wall clock synchronization"
        result = await extract_specifications(query)
        match result:
            case specs.Specifications(union=[specs.PaperSpec(content=str(content))]):
                assert content.lower() == "asynchronous wall clock synchronization"
            case _:
                raise ValueError(f"Unexpected result: {result}")

    async def test_extract_question_as_content(self) -> None:
        query = "What are some challenges of building multilingual evaluation datasets by translating existing English datasets into other target languages?"
        result = await extract_specifications(query)
        match result:
            case specs.Specifications(
                union=[
                    specs.PaperSpec(
                        name=None,
                        full_name=None,
                        field_of_study=None,
                        content=str(content),
                        years=None,
                        venue=None,
                        venue_group=None,
                        publication_type=None,
                        min_citations=None,
                        authors=None,
                        min_total_authors=None,
                        citing=None,
                        cited_by=None,
                        exclude=None,
                    )
                ]
            ):
                assert len(content) > 0
            case _:
                raise ValueError(f"Unexpected result: {result}")

    async def test_extract_content_with_metadata(self) -> None:
        query = "Randomized shortest path algorithms before 1980 but no by Dijkstra"
        result = await extract_specifications(query)
        match result:
            case specs.Specifications(
                union=[
                    specs.PaperSpec(
                        content=str(content),
                        years=specs.Years(end=1979),
                        exclude=specs.PaperSpec(authors=specs.AuthorSpec(name="Dijkstra")),
                    )
                ]
            ):
                assert len(content) > 0
            case _:
                raise ValueError(f"Unexpected result: {result}")
