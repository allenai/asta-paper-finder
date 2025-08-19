from dataclasses import dataclass

import pytest

from mabool.agents.metadata_only import ops
from mabool.agents.query_analyzer.test_extract_specs import Test as SpecTest
from mabool.data_model import specifications as specs

from .plan import plan


@dataclass
class Test(SpecTest):
    op: ops.DocOp


testdata: list[Test] = [
    Test(
        "By paper title only",
        'The "Attention is All You Need" paper',
        specs.PaperSpec(name="Attention is All You Need"),
        ops.FromS2ByTitle(name="Attention is All You Need"),
    ),
    Test(
        "By title and venue",
        "The 'Attention is All You Need' paper from NeurIPS",
        specs.PaperSpec(name="Attention is All You Need", venue="NeurIPS"),
        ops.FromS2ByTitle(name="Attention is All You Need", venues=["NeurIPS"]),
    ),
    Test(
        "By title and time range",
        "The 'Attention is All You Need' paper from 2017",
        specs.PaperSpec(name="Attention is All You Need", years=specs.Years(start=2017, end=2017)),
        ops.FromS2ByTitle(name="Attention is All You Need", time_range=specs.Years(start=2017, end=2017)),
    ),
    Test(
        "By title and several venues",
        "The 'Attention is All You Need' paper from NeurIPS or ICLR",
        specs.PaperSpec(name="Attention is All You Need", venue=specs.Set(op="or", items=["NeurIPS", "ICLR"])),
        ops.FromS2ByTitle(name="Attention is All You Need", venues=["NeurIPS", "ICLR"]),
    ),
    Test(
        "By title and several time ranges",
        "The 'Attention is All You Need' paper from 2017 or 2018",
        specs.PaperSpec(
            name="Attention is All You Need",
            years=specs.Set(op="or", items=[specs.Years(start=2017, end=2017), specs.Years(start=2018, end=2018)]),
        ),
        ops.Union(
            items=[
                ops.FromS2ByTitle(name="Attention is All You Need", time_range=specs.Years(start=2017, end=2017)),
                ops.FromS2ByTitle(name="Attention is All You Need", time_range=specs.Years(start=2018, end=2018)),
            ]
        ),
    ),
    Test(
        "By single author",
        "Papers by Gilad Bracha",
        specs.PaperSpec(authors=specs.AuthorSpec(name="Gilad Bracha")),
        ops.FromS2ByAuthorByName(author="Gilad Bracha"),
    ),
    Test(
        "By multiple authors - and",
        "Papers by Gilad Bracha and William Cook",
        specs.PaperSpec(
            authors=specs.Set(
                op="and", items=[specs.AuthorSpec(name="Gilad Bracha"), specs.AuthorSpec(name="William Cook")]
            )
        ),
        ops.Intersect(
            items=[ops.FromS2ByAuthorByName(author="Gilad Bracha"), ops.FromS2ByAuthorByName(author="William Cook")]
        ),
    ),
    Test(
        "By multiple authors - or",
        "Papers by Gilad Bracha or William Cook",
        specs.PaperSpec(
            authors=specs.Set(
                op="or", items=[specs.AuthorSpec(name="Gilad Bracha"), specs.AuthorSpec(name="William Cook")]
            )
        ),
        ops.Union(
            items=[ops.FromS2ByAuthorByName(author="Gilad Bracha"), ops.FromS2ByAuthorByName(author="William Cook")]
        ),
    ),
    Test(
        "By author of a paper",
        "Papers by authors of the PlayGo paper",
        specs.PaperSpec(
            authors=specs.AuthorSpec(papers=specs.PaperSet(op="any_author_of", items=[specs.PaperSpec(name="PlayGo")]))
        ),
        ops.ByAuthorsOfPapers(
            all_authors=False,
            authors=ops.AuthorsOfPapers(papers=ops.FromS2ByTitle(name="PlayGo", time_range=None, venues=None)),
        ),
    ),
    Test(
        "Cited by a paper",
        "Papers cited by the Attention paper",
        specs.PaperSpec(cited_by=specs.PaperSpec(name="Attention")),
        ops.GetAllReferences(source=ops.FromS2ByTitle(name="Attention", time_range=None, venues=None)),
    ),
    Test(
        "Cited by more than one paper",
        "papers cited by both the Attention paper and any paper by Guido Van Russom",
        specs.PaperSpec(
            cited_by=specs.Set(
                op="and",
                items=[
                    specs.PaperSpec(name="Attention"),
                    specs.PaperSpec(authors=specs.AuthorSpec(name="Guido Van Russom")),
                ],
            )
        ),
        ops.Intersect(
            items=[
                ops.GetAllReferences(
                    source=ops.FromS2ByTitle(
                        name="Attention",
                    )
                ),
                ops.GetAllReferences(source=ops.FromS2ByAuthorByName(author="Guido Van Russom")),
            ]
        ),
    ),
    Test(
        "One author citing another",
        "Papers by Dan Friedman citing Matthias Felleisen",
        specs.PaperSpec(
            authors=specs.AuthorSpec(name="Dan Friedman"),
            citing=specs.PaperSpec(authors=specs.AuthorSpec(name="Matthias Felleisen")),
        ),
        ops.FilterCiting(
            source=ops.EnrichWithReferences(source=ops.FromS2ByAuthorByName(author="Dan Friedman")),
            to_cite=ops.FromS2ByAuthorByName(author="Matthias Felleisen"),
        ),
    ),
    Test(
        "Citing a single paper spec",
        "Papers citing Aaron Turon",
        specs.PaperSpec(citing=specs.PaperSpec(authors=specs.AuthorSpec(name="Aaron Turon"))),
        ops.GetAllCiting(source=ops.FromS2ByAuthorByName(author="Aaron Turon")),
    ),
    Test(
        "Citing multiple paper specs",
        "Papers citing Aaron Turon or Guido Van Russom",
        specs.PaperSpec(
            citing=specs.Set(
                op="or",
                items=[
                    specs.PaperSpec(authors=specs.AuthorSpec(name="Aaron Turon")),
                    specs.PaperSpec(authors=specs.AuthorSpec(name="Guido Van Russom")),
                ],
            )
        ),
        ops.Union(
            items=[
                ops.GetAllCiting(source=ops.FromS2ByAuthorByName(author="Aaron Turon")),
                ops.GetAllCiting(source=ops.FromS2ByAuthorByName(author="Guido Van Russom")),
            ]
        ),
    ),
    Test(
        "Filter by metadata",
        "Paper cited by Andrew Ng, from 2010-2020, with more than 100 citations",
        specs.PaperSpec(
            cited_by=specs.PaperSpec(authors=specs.AuthorSpec(name="Andrew Ng")),
            years=specs.Years(start=2010, end=2020),
            min_citations=100,
        ),
        ops.FilterByMetadata(
            source=ops.GetAllReferences(source=ops.FromS2ByAuthorByName(author="Andrew Ng")),
            years=[specs.Years(start=2010, end=2020)],
            min_citations=100,
        ),
    ),
    Test(
        "Filter by metadata with or",
        "Papers by Andrew Ng at NeurIPS or ICCV with more than 100 citations",
        specs.PaperSpec(
            authors=specs.AuthorSpec(name="Andrew Ng"),
            venue=specs.Set(op="or", items=["NeurIPS", "ICCV"]),
            min_citations=100,
        ),
        ops.FilterByMetadata(
            source=ops.FromS2ByAuthorByName(author="Andrew Ng"), venue=["NeurIPS", "ICCV"], min_citations=100
        ),
    ),
    Test(
        "Find by full name",
        "The DistilBERT paper",
        specs.PaperSpec(
            name="DistilBERT", full_name="DistilBERT: A distilled version of BERT: smaller, faster, cheaper and lighter"
        ),
        ops.FromS2ByTitle(name="DistilBERT: A distilled version of BERT: smaller, faster, cheaper and lighter"),
    ),
    Test(
        "Cited by full name",
        "EMNLP papers cited by the RoBERTa paper",
        specs.PaperSpec(
            venue="EMNLP",
            cited_by=specs.PaperSpec(
                name="RoBERTa", full_name="RoBERTa: A Robustly Optimized BERT Pretraining Approach"
            ),
        ),
        ops.FilterByMetadata(
            venue=["EMNLP"],
            source=ops.GetAllReferences(
                source=ops.FromS2ByTitle(name="RoBERTa: A Robustly Optimized BERT Pretraining Approach")
            ),
        ),
    ),
    Test(
        "Filter by highly cited",
        "Highly cited papers by Glen Koch",
        specs.PaperSpec(authors=specs.AuthorSpec(name="Glen Koch"), min_citations="high"),
        ops.FilterByHighlyCited(source=ops.FromS2ByAuthorByName(author="Glen Koch")),
    ),
    Test(
        "Filter by highly cited and more",
        "Highly cited journal papers by Glen Koch",
        specs.PaperSpec(
            authors=specs.AuthorSpec(name="Glen Koch"), min_citations="high", publication_type="JournalArticle"
        ),
        ops.FilterByHighlyCited(
            source=ops.FilterByMetadata(
                source=ops.FromS2ByAuthorByName(author="Glen Koch"), publication_types=["JournalArticle"]
            )
        ),
    ),
    Test(
        "Filter by single cited by",
        "Papers by Sebastian Ruder cited by the T5 paper",
        specs.PaperSpec(
            authors=specs.AuthorSpec(name="Sebastian Ruder"),
            cited_by=specs.PaperSpec(
                name="T5", full_name="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
            ),
        ),
        ops.FilterCitedBy(
            source=ops.FromS2ByAuthorByName(author="Sebastian Ruder"),
            that_cite=ops.EnrichWithReferences(
                source=ops.FromS2ByTitle(
                    name="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
                )
            ),
        ),
    ),
    Test(
        "Filter by multiple cited by",
        "Papers by Sebastian Ruder cited by the T5 or RoBERTa paper",
        specs.PaperSpec(
            authors=specs.AuthorSpec(name="Sebastian Ruder"),
            cited_by=specs.Set(
                op="or",
                items=[
                    specs.PaperSpec(
                        name="T5",
                        full_name="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
                    ),
                    specs.PaperSpec(full_name="RoBERTa: A Robustly Optimized BERT Pretraining Approach"),
                ],
            ),
        ),
        ops.Union(
            items=[
                ops.FilterCitedBy(
                    source=ops.FromS2ByAuthorByName(author="Sebastian Ruder"),
                    that_cite=ops.EnrichWithReferences(
                        source=ops.FromS2ByTitle(
                            name="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
                        )
                    ),
                ),
                ops.FilterCitedBy(
                    source=ops.FromS2ByAuthorByName(author="Sebastian Ruder"),
                    that_cite=ops.EnrichWithReferences(
                        source=ops.FromS2ByTitle(name="RoBERTa: A Robustly Optimized BERT Pretraining Approach")
                    ),
                ),
            ]
        ),
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
                            name="BERT", full_name="Bidirectional Encoder Representations from Transformers"
                        )
                    ],
                ),
                min_authors=2,
            ),
        ),
        ops.FilterByMetadata(
            source=ops.ByAuthorsOfPapers(
                all_authors=False,
                authors=ops.AuthorsOfPapers(
                    papers=ops.FromS2ByTitle(name="Bidirectional Encoder Representations from Transformers")
                ),
                min_authors_of_papers=2,
            ),
            venue=["NAACL"],
        ),
    ),
    Test(
        "Author and any coauthor",
        "Papers by Mayer Goldberg and any coauthor",
        specs.PaperSpec(authors=specs.AuthorSpec(name="Mayer Goldberg"), min_total_authors=2),
        ops.FilterByMinTotalAuthors(source=ops.FromS2ByAuthorByName(author="Mayer Goldberg"), min_total_authors=2),
    ),
    Test(
        "Fetch by FoS filter by venue group",
        "IEEE biology papers from 2010 to 2015 with more than 100 citations",
        specs.PaperSpec(
            venue_group="IEEE", field_of_study="biology", years=specs.Years(start=2010, end=2015), min_citations=100
        ),
        ops.FilterByMetadata(
            source=ops.FromS2Search(
                fields_of_study=["Biology"], time_range=specs.Years(start=2010, end=2015), min_citations=100
            ),
            venue_group=["IEEE"],
        ),
    ),
    Test(
        "Papers by X but not by Y",
        "papers by Andrej Karpathy but not by Justin Johnson",
        specs.PaperSpec(
            authors=specs.AuthorSpec(name="Andrej Karpathy"),
            exclude=specs.PaperSpec(authors=specs.AuthorSpec(name="Justin Johnson")),
        ),
        ops.FilterExclude(
            source=ops.FromS2ByAuthorByName(author="Andrej Karpathy"),
            author=ops.FindAuthorByName(author="Justin Johnson"),
        ),
    ),
    Test(
        "Papers citing X but not Y",
        "Papers citing the Sentence T5 paper but not citing the RoBERTa paper",
        specs.PaperSpec(
            citing=specs.PaperSpec(
                name="Sentence T5",
                full_name="Sentence T5: Scalable Sentence Pre-training for Natural Language Generation",
            ),
            exclude=specs.PaperSpec(
                citing=specs.PaperSpec(
                    name="RoBERTa", full_name="RoBERTa: A Robustly Optimized BERT Pretraining Approach"
                )
            ),
        ),
        ops.FilterExclude(
            source=ops.EnrichWithReferences(
                source=ops.GetAllCiting(
                    source=ops.FromS2ByTitle(
                        name="Sentence T5: Scalable Sentence Pre-training for Natural Language Generation"
                    )
                )
            ),
            citing=ops.FromS2ByTitle(name="RoBERTa: A Robustly Optimized BERT Pretraining Approach"),
        ),
    ),
]


@pytest.mark.parametrize("test", testdata, ids=Test.id)
def test_plan(test: Test) -> None:
    spec = specs.Specifications(union=[test.spec])
    result = plan(spec)
    assert result == test.op


def test_filter_by_citing_multiple_specs() -> None:
    'Papers by Furu Wei citing the "Attention is All You Need" paper and Guido Van Russom'
    spec = specs.Specifications(
        union=[
            specs.PaperSpec(
                authors=specs.AuthorSpec(name="Furu Wei"),
                citing=specs.Set(
                    op="and",
                    items=[
                        specs.PaperSpec(name="Attention"),
                        specs.PaperSpec(authors=specs.AuthorSpec(name="Guido Van Russom")),
                    ],
                ),
            )
        ]
    )
    enrich_refs = ops.EnrichWithReferences(source=ops.FromS2ByAuthorByName(author="Furu Wei"))
    expected = ops.Intersect(
        items=[
            ops.FilterCiting(source=enrich_refs, to_cite=ops.FromS2ByTitle(name="Attention")),
            ops.FilterCiting(source=enrich_refs, to_cite=ops.FromS2ByAuthorByName(author="Guido Van Russom")),
        ]
    )
    result = plan(spec)
    assert result == expected
    match result:
        case ops.Intersect(items=[ops.FilterCiting(source=s1), ops.FilterCiting(source=s2)]):
            assert s1 is s2
        case _:
            raise AssertionError("Unexpected result structure")
