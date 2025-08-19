from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Self, TypeVar, final

from ai2i.dcollection.interface.document import ExtractedYearlyTimeRange

from mabool.data_model import specifications as specs

from . import ops


def resolve_author(author_spec: specs.AuthorSpec) -> ops.DocOp:
    match author_spec:
        case specs.AuthorSpec(name=str(name), papers=None):
            return ops.FromS2ByAuthorByName(author=name)
        case specs.AuthorSpec(affiliation=affiliation) if affiliation:
            raise NotImplementedError("Author affiliation is not supported yet")
        case specs.AuthorSpec(
            papers=specs.PaperSet(
                op=op,
                items=[paper_spec],
            ),
            min_authors=min_authors,
        ):
            prev = resolve_paper(paper_spec)
            authors_of_papers = ops.AuthorsOfPapers(papers=prev)
            return ops.ByAuthorsOfPapers(
                authors=authors_of_papers, all_authors=(op == "all_authors_of"), min_authors_of_papers=min_authors
            )
        case specs.AuthorSpec(papers=paper_specs) if paper_specs:
            raise NotImplementedError("Author papers are not supported yet")
        case _:
            raise ValueError("Invalid author specification")


class Rule(ABC):
    rules: list[Self] = []

    @abstractmethod
    def consumes(self) -> list[str]:
        """
        Abstract method to define the consumed attributes.
        """
        pass

    def __init_subclass__(cls, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init_subclass__(*args, **kwargs)
        if cls is not Rule:
            Rule.rules.append(cls())

    def resolve(self, paper_spec: specs.PaperSpec, prev: Optional[ops.DocOp]) -> Optional[ops.DocOp]:
        """
        Resolve a part of the paper specification into a plan.
        :param paper_spec: The paper specification to resolve.
        :param prev: The previous plan, if any.
        :return: The resolved plan.
        """
        match (self, prev):
            case (Source() as source, prev):
                plan = source.apply(paper_spec)
                if plan:
                    if prev:
                        return ops.Intersect(items=[prev, plan])
                    return plan
            case (Primary() as primary, None):
                return primary.generate(paper_spec)
            case (Filter() as filter, ops.DocOp()):
                return filter.filter(paper_spec, prev)
            case _:
                return None


class Source(Rule, ABC):
    @abstractmethod
    def apply(self, paper_spec: specs.PaperSpec) -> Optional[ops.DocOp]:
        """
        Compute a plan from the paper specification.
        :param paper_spec: The paper specification to resolve.
        :return: The resolved plan.
        """
        pass


class Primary(Rule, ABC):
    @abstractmethod
    def generate(self, paper_spec: specs.PaperSpec) -> Optional[ops.DocOp]:
        """
        Generate a plan from the paper specification.
        :param paper_spec: The paper specification to resolve.
        :return: The resolved plan.
        """
        pass


class Filter(Rule, ABC):
    @abstractmethod
    def filter(self, paper_spec: specs.PaperSpec, prev: Optional[ops.DocOp]) -> Optional[ops.DocOp]:
        """
        Filter a plan from the paper specification.
        :param paper_spec: The paper specification to resolve.
        :param prev: The previous plan, if any.
        :return: The resolved plan.
        """
        pass


@final
class GenByContent(Source):
    def consumes(self) -> list[str]:
        return ["content"]

    def apply(self, paper_spec: specs.PaperSpec) -> Optional[ops.Plan]:
        match paper_spec:
            case specs.PaperSpec(content=content) if content:
                return ops.Plan(action=f"Search: {content}")
            case _:
                return None


@final
class GenByNameVenueYears(Source):
    def consumes(self) -> list[str]:
        return ["name", "full_name", "venue", "years"]

    def apply(self, paper_spec: specs.PaperSpec) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(name=name, full_name=full_name, venue=venue, years=years) if full_name or name:
                title = full_name or name
                assert title
                match venue:
                    case None:
                        venues = None
                    case str():
                        venues = [venue]
                    case specs.Set(op="or", items=items):
                        venues = items
                    case specs.Set(op="and", items=items):
                        raise ValueError("AND operation is not supported for venues")
                    case _:
                        raise ValueError("Invalid venue specification")
                match years:
                    case ExtractedYearlyTimeRange() | None:
                        return ops.FromS2ByTitle(name=title, time_range=years, venues=venues)
                    case specs.Set(op="or", items=items):
                        deps = list(map(lambda x: ops.FromS2ByTitle(name=title, time_range=x, venues=venues), items))
                        return ops.Union(items=deps)
                    case specs.Set(op="and"):
                        raise ValueError("AND operation is not supported for years")
                    case _:
                        raise ValueError("Invalid years specification")
            case _:
                return None


@final
class GenByVenueYears(Source):
    def consumes(self) -> list[str]:
        return ["venue", "years"]

    def apply(self, paper_spec: specs.PaperSpec) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(venue=venue, years=years) if venue and years:
                match venue:
                    case str():
                        venues = [venue]
                    case specs.Set(op="or", items=items):
                        venues = items
                    case specs.Set(op="and", items=items):
                        raise ValueError("AND operation is not supported for venues")
                    case _:
                        raise ValueError("Invalid venue specification")
                match years:
                    case ExtractedYearlyTimeRange() | None:
                        return ops.FromS2Search(venues=venues, time_range=years)
                    case specs.Set(op="or", items=items):
                        deps = list(map(lambda x: ops.FromS2Search(venues=venues, time_range=x), items))
                        return ops.Union(items=deps)
                    case specs.Set(op="and"):
                        raise ValueError("AND operation is not supported for years")
                    case _:
                        raise ValueError("Invalid years specification")
            case _:
                return None


@final
class GenByAuthors(Source):
    def consumes(self) -> list[str]:
        return ["authors"]

    def apply(self, paper_spec: specs.PaperSpec) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(authors=specs.AuthorSpec() as author_spec):
                return resolve_author(author_spec)
            case specs.PaperSpec(authors=specs.Set(items=items, op=op)):
                authors = list(map(resolve_author, items))
                match op:
                    case "and":
                        action = ops.Intersect
                    case "or":
                        action = ops.Union
                return action(items=authors)
            case _:
                return None


@final
class PrimeByCitedBySinglePaperSpec(Primary):
    def consumes(self) -> list[str]:
        return ["cited_by"]

    def generate(self, paper_spec: specs.PaperSpec) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(cited_by=specs.PaperSpec() as papers_citing_target):
                find_papers_citing = resolve_paper(papers_citing_target)
                return ops.GetAllReferences(source=find_papers_citing)
            case _:
                return None


@final
class PrimeByCitedByMultiplePaperSpecs(Primary):
    def consumes(self) -> list[str]:
        return ["cited_by"]

    def generate(self, paper_spec: specs.PaperSpec) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(cited_by=specs.Set(op=op, items=items)):
                find_all_papers_citing = list(map(resolve_paper, items))
                get_all_refs = list(map(lambda x: ops.GetAllReferences(source=x), find_all_papers_citing))
                match op:
                    case "and":
                        action = ops.Intersect
                    case "or":
                        action = ops.Union
                return action(items=get_all_refs)
            case _:
                return None


@final
class FilterByCitingSinglePaperSpec(Filter):
    def consumes(self) -> list[str]:
        return ["citing"]

    def filter(self, paper_spec: specs.PaperSpec, prev: Optional[ops.DocOp]) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(citing=specs.PaperSpec() as papers_cited_by_target) if prev:
                find_papers_cited = resolve_paper(papers_cited_by_target)
                get_refs = ops.EnrichWithReferences(source=prev)
                return ops.FilterCiting(source=get_refs, to_cite=find_papers_cited)
            case _:
                return None


@final
class FilterByCitingMultiplePaperSpecs(Filter):
    def consumes(self) -> list[str]:
        return ["citing"]

    def filter(self, paper_spec: specs.PaperSpec, prev: Optional[ops.DocOp]) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(citing=specs.Set(op=op, items=items)) if prev:
                find_all_papers_cited_by = list(map(resolve_paper, items))
                enrich_refs = ops.EnrichWithReferences(source=prev)
                filter_by_citing = list(
                    map(lambda x: ops.FilterCiting(source=enrich_refs, to_cite=x), find_all_papers_cited_by)
                )
                op_cls = ops.Intersect if op == "and" else ops.Union
                return op_cls(items=filter_by_citing)
            case _:
                return None


@final
class FilterByCitedBySinglePaperSpec(Filter):
    def consumes(self) -> list[str]:
        return ["cited_by"]

    def filter(self, paper_spec: specs.PaperSpec, prev: Optional[ops.DocOp]) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(cited_by=specs.PaperSpec() as papers_citing_target) if prev:
                find_papers_citing = resolve_paper(papers_citing_target)
                get_refs = ops.EnrichWithReferences(source=find_papers_citing)
                return ops.FilterCitedBy(source=prev, that_cite=get_refs)
            case _:
                return None


@final
class FilterByCitedByMultiplePaperSpecs(Filter):
    def consumes(self) -> list[str]:
        return ["cited_by"]

    def filter(self, paper_spec: specs.PaperSpec, prev: Optional[ops.DocOp]) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(cited_by=specs.Set(op=op, items=items)) if prev:
                find_all_papers_citing = list(map(resolve_paper, items))
                filter_by_cited_by = list(
                    map(
                        lambda x: ops.FilterCitedBy(source=prev, that_cite=ops.EnrichWithReferences(source=x)),
                        find_all_papers_citing,
                    )
                )
                op_cls = ops.Intersect if op == "and" else ops.Union
                return op_cls(items=filter_by_cited_by)
            case _:
                return None


@final
class PrimeByCitingSinglePaper(Primary):
    def consumes(self) -> list[str]:
        return ["citing"]

    def generate(self, paper_spec: specs.PaperSpec) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(citing=specs.PaperSpec() as papers_cited_by_target):
                find_papers_cited = resolve_paper(papers_cited_by_target)
                return ops.GetAllCiting(source=find_papers_cited)
            case _:
                return None


@final
class PrimeByCitingMultiplePapers(Primary):
    def consumes(self) -> list[str]:
        return ["citing"]

    def generate(self, paper_spec: specs.PaperSpec) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(citing=specs.Set(items=items, op=op)):
                find_all_papers_cited_by = list(map(resolve_paper, items))
                get_all_citations = list(map(lambda x: ops.GetAllCiting(source=x), find_all_papers_cited_by))
                op_cls = ops.Intersect if op == "and" else ops.Union
                return op_cls(items=get_all_citations)
            case _:
                return None


@final
class PrimeByS2Search(Primary):
    def consumes(self) -> list[str]:
        return ["years", "venue", "field_of_study", "min_citations"]

    def generate(self, paper_spec: specs.PaperSpec) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(
                years=years, venue=venue, field_of_study=field_of_study, min_citations=min_citations
            ) if venue or field_of_study:
                match venue:
                    case str():
                        venues = [venue]
                    case specs.Set(op="or", items=items):
                        venues = items
                    case specs.Set(op="and", items=items):
                        raise ValueError("AND operation is not supported for venues")
                    case _:
                        venues = None
                match field_of_study:
                    case str(fields_of_study):
                        fields_of_study = [fields_of_study]
                    case None:
                        fields_of_study = None
                f = None
                match min_citations:
                    case int() as min_citations:
                        if min_citations < 0:
                            raise ValueError("Minimum citations must be a non-negative integer")
                        f = lambda y: ops.FromS2Search(
                            venues=venues, time_range=y, fields_of_study=fields_of_study, min_citations=min_citations
                        )
                    case "high":
                        f = lambda y: ops.FilterByHighlyCited(
                            source=ops.FromS2Search(venues=venues, time_range=y, fields_of_study=fields_of_study)
                        )
                    case None:
                        f = lambda y: ops.FromS2Search(venues=venues, time_range=y, fields_of_study=fields_of_study)
                assert f
                match years:
                    case ExtractedYearlyTimeRange():
                        return f(years)
                    case specs.Set(op="or", items=items):
                        deps = list(map(lambda x: f(x), items))
                        return ops.Union(items=deps)
                    case specs.Set(op="and"):
                        raise ValueError("AND operation is not supported for years")
                    case None:
                        return ops.FromS2Search(venues=venues)
                    case _:
                        return None
            case _:
                return None


@final
class FilterByMinTotalAuthors(Filter):
    def consumes(self) -> list[str]:
        return ["min_total_authors"]

    def filter(self, paper_spec: specs.PaperSpec, prev: Optional[ops.DocOp]) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(min_total_authors=min_total_authors) if min_total_authors and prev:
                return ops.FilterByMinTotalAuthors(source=prev, min_total_authors=min_total_authors)
            case _:
                return None


@final
class FilterByMetadata(Filter):
    def consumes(self) -> list[str]:
        return ["years", "venue", "venue_group", "field_of_study", "publication_type", "min_citations"]

    T = TypeVar("T")

    @staticmethod
    def _norm(c: specs.Composite[T]) -> list[T] | None:
        match c:
            case None:
                return None
            case specs.Set(op="or", items=items):
                return items
            case specs.Set(op="and", items=items):
                raise ValueError("AND operation is not supported for metadata")
            case ExtractedYearlyTimeRange() | str() as item:
                return [item]
            case _:
                raise ValueError("Invalid metadata specification")

    def filter(self, paper_spec: specs.PaperSpec, prev: Optional[ops.DocOp]) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(
                years=years,
                venue=venue,
                venue_group=venue_group,
                field_of_study=field_of_study,
                publication_type=publication_type,
                min_citations=min_citations,
            ) if any((years, venue, venue_group, field_of_study, publication_type, min_citations)) and prev:
                years = self._norm(years)
                venue = self._norm(venue)
                venue_group = self._norm(venue_group)
                publication_type = self._norm(publication_type)
                match min_citations:
                    case "high":
                        if not any((years, venue, publication_type)):
                            return ops.FilterByHighlyCited(source=prev)
                        else:
                            my_filter = ops.FilterByMetadata(
                                source=prev,
                                years=years,
                                venue=venue,
                                venue_group=venue_group,
                                field_of_study=field_of_study,
                                publication_types=publication_type,
                            )
                            return ops.FilterByHighlyCited(source=my_filter)
                    case int() | None as min_citations:
                        pass
                return ops.FilterByMetadata(
                    source=prev,
                    years=years,
                    venue=venue,
                    venue_group=venue_group,
                    field_of_study=field_of_study,
                    publication_types=publication_type,
                    min_citations=min_citations,
                )
            case _:
                return None


@final
class FilterExclude(Filter):
    def consumes(self) -> list[str]:
        return ["exclude"]

    def filter(self, paper_spec: specs.PaperSpec, prev: Optional[ops.DocOp]) -> Optional[ops.DocOp]:
        match paper_spec:
            case specs.PaperSpec(exclude=exclude) if exclude and prev:
                match exclude:
                    case specs.PaperSpec(
                        years=years,
                        venue=venue,
                        venue_group=venue_group,
                        field_of_study=field_of_study,
                        publication_type=publication_type,
                        min_citations=int() | None as min_citations,
                        authors=specs.AuthorSpec() | None as author,
                        citing=specs.PaperSpec() | None as citing,
                        cited_by=specs.PaperSpec() | None as cited_by,
                    ):
                        source = ops.EnrichWithReferences(source=prev) if citing else prev
                        years = FilterByMetadata._norm(years)
                        venue = FilterByMetadata._norm(venue)
                        venue_group = FilterByMetadata._norm(venue_group)
                        publication_type = FilterByMetadata._norm(publication_type)
                        author_op = None
                        match author:
                            case specs.AuthorSpec(name=str(author_name)):
                                author_op = ops.FindAuthorByName(author=author_name)
                            case _:
                                pass
                        citing_op = resolve_paper(citing) if citing else None
                        cited_by_op = resolve_paper(cited_by) if cited_by else None
                        return ops.FilterExclude(
                            source=source,
                            years=years,
                            venue=venue,
                            venue_group=venue_group,
                            field_of_study=field_of_study,
                            publication_types=publication_type,
                            min_citations=min_citations,
                            author=author_op,
                            citing=citing_op,
                            cited_by=cited_by_op,
                        )
                    case _:
                        return None
            case _:
                return None


def resolve_paper(paper_spec: specs.PaperSpec, prev: Optional[ops.DocOp] = None) -> ops.DocOp:
    if paper_spec.is_empty():
        if prev:
            return prev
        raise ValueError("Empty paper specification")
    for rule in Rule.rules:
        plan = rule.resolve(paper_spec, prev)
        if plan:
            remain = paper_spec.delete(*rule.consumes())
            # if rule is primary and was applied, prev is None, and step is act
            return resolve_paper(remain, plan)
    # If no rules matched, raise an error
    raise NotImplementedError("No rules matched for paper specification")


def plan(specifications: specs.Specifications) -> ops.DocOp:
    match specifications:
        case specs.Specifications(union=[]):
            raise ValueError("No specifications provided")
        case specs.Specifications(union=[paper_spec]):
            return resolve_paper(paper_spec, prev=None)
        case specs.Specifications(union=paper_specs):
            items = list(map(resolve_paper, paper_specs))
            return ops.Union(items=items)
