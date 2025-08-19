import logging
from datetime import datetime
from typing import Any, Coroutine

from ai2i.chain import LLMEndpoint, LLMModel, Timeouts, define_llm_endpoint
from ai2i.config import config_value
from ai2i.dcollection import (
    DocumentCollection,
    DocumentCollectionFactory,
    DocumentCollectionSortDef,
    ExtractedYearlyTimeRange,
    s2_get_authors_by_name,
)
from ai2i.di import DI
from semanticscholar.Author import Author

from mabool.agents.broad_search_by_keyword.broad_search_by_keyword_agent import (
    suggest_retrieval_query,
)
from mabool.agents.common.common import (
    AgentState,
    filter_by_author,
    filter_docs_by_metadata,
)
from mabool.agents.common.computed_fields.fields import rerank_score_field
from mabool.agents.common.computed_fields.relevance import relevance_judgement_field
from mabool.agents.common.domain_utils import get_fields_of_study_filter_from_domains
from mabool.agents.common.relevance_judgement_utils import get_relevant_docs
from mabool.agents.common.utils import alog_args
from mabool.agents.llm_suggestion.llm_suggestion_agent import get_llm_suggested_papers
from mabool.agents.search_by_authors.search_by_authors_prompts import (
    disambiguate_user_response,
)
from mabool.data_model.agent import (
    AgentError,
    AgentInput,
    AgentOutput,
    BroadOrSpecificLiterals,
    DomainsIdentified,
    RelevanceCriteria,
)
from mabool.data_model.config import cfg_schema
from mabool.infra.operatives import (
    CompleteResponse,
    InquiryQuestion,
    Operative,
    OperativeResponse,
    VoidResponse,
)
from mabool.utils import dc_deps
from mabool.utils.asyncio import custom_gather
from mabool.utils.dc import DC
from mabool.utils.llm_utils import get_api_key_for_model

logger = logging.getLogger(__name__)


class SearchByAuthorsInput(AgentInput):
    authors: list[str]
    broad_or_specific: BroadOrSpecificLiterals
    user_content_input: str | None = None
    relevance_criteria: RelevanceCriteria | None = None
    time_range: ExtractedYearlyTimeRange | None = None
    venues: list[str] | None = None
    domains: DomainsIdentified


class NoAuthorMatchedError(Exception):
    message: str

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.message = msg


SearchByAuthorsState = AgentState
SearchByAuthorsOutput = AgentOutput

AUTHOR_PAGINATION = 5
prefix = "Please specify which of the following you are interested in (out of {amount}):\n\n"
author_candidate = '{idx}. <Author authorId="{id}">{name}</Author> - {count} papers, {h_index} h-index.'  # noqa: E501
suffix = "\n\nPlease type the index of the author (can be an index also from previous responds)"
next_suffix = ', or "next" otherwise.'
no_next_suffix = " as this is the last batch of authors."


def get_default_endpoint() -> LLMEndpoint:
    llm_model = LLMModel.from_name(config_value(cfg_schema.search_by_author_agent.llm_model_name))
    return define_llm_endpoint(
        default_timeout=Timeouts.medium,
        default_model=llm_model,
        logger=logger,
        api_key=get_api_key_for_model(llm_model),
    )


def get_time_range_hint(time_range: ExtractedYearlyTimeRange | None) -> str:
    if not time_range:
        return ""

    if time_range.start:
        if time_range.start == time_range.end:
            return f"During {time_range.start}."
        else:
            return (
                f"Sometime between {time_range.start}"
                + f" and {time_range.end if time_range.end else datetime.now().year}."
            )
    else:
        if time_range.end:
            return f"Sometime no later than {time_range.end}."

    return ""


def get_venues_hint(venues: list[str] | None) -> str:
    if venues:
        return f"Published in {' or '.join(venues)}."

    return ""


class SearchByAuthorsAgent(Operative[SearchByAuthorsInput, SearchByAuthorsOutput, SearchByAuthorsState]):
    def register(self) -> None: ...

    async def disambiguate_author(self, author_name: str, found_authors: list[Author]) -> list[Author]:
        sorted_authors = sorted(
            found_authors,
            key=lambda a: (
                # if more than just last/first name, check for exact match (case insensitive)
                len(author_name.strip().split()) > 1 and a.name.lower() == author_name.lower(),
                # check if all parts of the name are in the author name (to deprioritize initials)
                all(name_part.lower() in a.name.lower().split() for name_part in author_name.split()),
                int(a.hIndex),
                int(a.paperCount),
            ),
            reverse=True,
        )
        inquiry = self.inquiry()
        if (
            not config_value(cfg_schema.search_by_author_agent.disambiguate_authors)
            or inquiry is None
            or len(sorted_authors) == 1
        ):
            return sorted_authors[: config_value(cfg_schema.search_by_author_agent.consider_profiles_per_author)]

        for i in range(0, len(sorted_authors), AUTHOR_PAGINATION):
            not_last_batch = i + AUTHOR_PAGINATION < len(sorted_authors)
            cur_authors = sorted_authors[i : i + AUTHOR_PAGINATION]
            options = [str(i + j + 1) for j in range(len(cur_authors))] + (["next"] if not_last_batch else [])
            formatted_candidates = [
                author_candidate.format(
                    idx=i + j + 1,
                    name=a.name,
                    count=a.paperCount,
                    id=a.authorId,
                    h_index=a.hIndex,
                )
                for j, a in enumerate(cur_authors)
            ]
            formulated_question = (
                prefix.format(amount=len(sorted_authors))
                + "\n".join(formatted_candidates)
                + suffix
                + (next_suffix if not_last_batch else no_next_suffix)
            )

            inquire_response = await inquiry.ask(InquiryQuestion(question=formulated_question, options=options))
            try:
                if inquire_response.answer == "next":
                    continue
                # best effort, if its already int-able convert it, otherwise try to disambiguate
                disambiguated_user_response = int(inquire_response.answer)
                return [sorted_authors[disambiguated_user_response - 1]]
            except ValueError:
                options_without_next = options[:-1] if "next" == options[-1] else options
                disambiguated_user_response = (
                    await get_default_endpoint()
                    .execute(disambiguate_user_response)
                    .once(
                        {
                            "agents_question": formulated_question,
                            "options": options_without_next,
                            "user_response": inquire_response.answer,
                        }
                    )
                )
                return [sorted_authors[disambiguated_user_response - 1]]

        raise NoAuthorMatchedError("There are no more authors matching this name")

    @DI.managed
    async def get_authors_papers_by_s2_authors(
        self,
        authors: list[str],
        doc_collection_factory: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory),
    ) -> DocumentCollection:
        # get top authors by paperCount per requested author
        top_authors = []
        for author in authors:
            found_authors = await s2_get_authors_by_name(author, doc_collection_factory.context())
            if not found_authors:
                raise NoAuthorMatchedError(f"No author was found by this name: {author}")
            top_authors.append(await self.disambiguate_author(author, found_authors))

        flat_authors = [a for author_profile in top_authors for a in author_profile]

        search_results = await DC.from_s2_by_author(
            [flat_authors],
            config_value(cfg_schema.s2_api.total_papers_limit),
        )
        if len(authors) > 1:
            # here we keep only the papers that are actually with all the authors,
            #   but since we have k profiles per author we check at least one of each author's profile group appears
            # NOTE - this could be done with co_author api but it currently it not efficient
            search_results = search_results.filter(
                lambda doc: all(
                    [
                        len(
                            {a.author_id for a in (doc.authors if doc.authors else [])}.intersection(
                                {a.authorId for a in author_profiles}
                            )
                        )
                        for author_profiles in top_authors
                    ]
                )
            )

        return search_results

    async def get_authors_papers_by_s2_relevance(
        self,
        authors: list[str],
        user_content_input: str,
        domains: DomainsIdentified,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
    ) -> DocumentCollection:
        reformulated_query = await suggest_retrieval_query(user_content_input)
        results = await DC.from_s2_search(
            query=" ".join([reformulated_query] + authors),
            limit=config_value(cfg_schema.search_by_author_agent.relevance_judgements_quota),
            time_range=time_range,
            venues=venues,
            fields_of_study=get_fields_of_study_filter_from_domains(domains),
        )

        # filter results by author matches
        results = results.filter(lambda doc: filter_by_author(authors, doc, keep_missing=False))
        return results

    def get_authors_papers_fast_and_naive_methods(
        self,
        authors: list[str],
        user_content_input: str,
        domains: DomainsIdentified,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
    ) -> list[Coroutine[Any, Any, DocumentCollection]]:
        futures = []

        futures.append(
            self.get_authors_papers_by_s2_relevance(authors, user_content_input, domains, time_range, venues)
        )

        extra_hints = [
            f"The paper was written by {' and '.join(authors)}.",
            get_time_range_hint(time_range),
            get_venues_hint(venues),
        ]
        futures.append(
            get_llm_suggested_papers(
                user_input=user_content_input,
                domains=domains,
                extra_hints=" ".join(filter(None, extra_hints)),
            )
        )

        return futures

    @DI.managed
    async def relevance(
        self,
        results: DocumentCollection,
        user_content_input: str | None,
        relevance_criteria: RelevanceCriteria | None,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
    ) -> DocumentCollection:
        if (
            (time_range is not None and time_range.non_empty())
            or (venues is not None and len(venues) > 1)
            or (
                user_content_input
                and relevance_criteria is not None
                and relevance_criteria.required_relevance_critieria
            )
        ):
            results = await filter_docs_by_metadata(
                results,
                time_range,
                venues,
                keep_missing=False,
            )

            if (
                user_content_input
                and relevance_criteria is not None
                and relevance_criteria.required_relevance_critieria
            ):
                quota = config_value(cfg_schema.search_by_author_agent.relevance_judgements_quota)
                if len(results.documents) > quota:
                    results = await results.with_fields([rerank_score_field(relevance_criteria)])
                    results = results.sorted([DocumentCollectionSortDef(field_name="rerank_score", order="desc")]).take(
                        quota
                    )
                results = await results.with_fields([relevance_judgement_field(relevance_criteria)])
                results = get_relevant_docs(results)
        return results

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self,
        state: SearchByAuthorsState | None,
        inputs: SearchByAuthorsInput,
    ) -> tuple[SearchByAuthorsState | None, OperativeResponse[SearchByAuthorsOutput]]:
        try:
            futures = []

            # if we have content lets try some more "naive" and fast methods:
            #   1. with s2 relevance
            #   2. with llm suggest
            if inputs.user_content_input:
                futures.extend(
                    self.get_authors_papers_fast_and_naive_methods(
                        inputs.authors,
                        inputs.user_content_input,
                        inputs.domains,
                        inputs.time_range,
                        inputs.venues,
                    )
                )

            futures.append(self.get_authors_papers_by_s2_authors(inputs.authors))

            result_sets = await custom_gather(*futures, return_exceptions=True)
            no_exceptions_result_sets = []
            unknown_exceptions = []
            for result_set in result_sets:
                if isinstance(result_set, NoAuthorMatchedError):
                    response_text = result_set.message
                    return (
                        state,
                        CompleteResponse(
                            data=SearchByAuthorsOutput(response_text=response_text, doc_collection=DC.empty()),
                        ),
                    )
                if isinstance(result_set, BaseException):
                    unknown_exceptions.append(result_set)
                    continue
                no_exceptions_result_sets.append(result_set)
            if len(no_exceptions_result_sets) == 0:
                raise unknown_exceptions[0]

            logger.info("All results gathered, merging...")
            results: DocumentCollection = no_exceptions_result_sets[0]
            for no_exceptions_result_set in no_exceptions_result_sets[1:]:
                results += no_exceptions_result_set

            results = await self.relevance(
                results,
                inputs.user_content_input,
                inputs.relevance_criteria,
                inputs.time_range,
                inputs.venues,
            )

            # here we assume its a specific query so we need to return only a few results
            if inputs.broad_or_specific == "specific":
                results = results.take(config_value(cfg_schema.search_by_author_agent.limit_for_specific))

            # NOTE - the results are not ordered in any meaningful way, sorting will be done by caller

        except Exception as e:
            return None, VoidResponse(error=AgentError(type="other", message=str(e)))

        return (
            state,
            CompleteResponse(data=SearchByAuthorsOutput(response_text="", doc_collection=results)),
        )
