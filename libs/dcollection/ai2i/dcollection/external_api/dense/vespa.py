from __future__ import annotations

import logging
from typing import Any, TypedDict

import httpx
from httpx import NetworkError, Response, TimeoutException
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from semanticscholar import AsyncSemanticScholar

from ai2i.dcollection.interface.document import (
    Author,
    BoundingBox,
    CorpusId,
    ExtractedYearlyTimeRange,
)

logger = logging.getLogger(__name__)


MAX_TOP_RESULTS = 400


class Span(TypedDict):
    start: int
    end: int


class RefMention(Span):
    matchedPaperCorpusId: str


class SnippetAnnotations(TypedDict):
    sentences: list[Span]
    refMentions: list[RefMention]


class ExceedQuotaError(Exception):
    """Too Many Requests. Please wait and try again or apply for a key for higher rate limits."""


def build_filters(
    time_range: ExtractedYearlyTimeRange | None = None,
    venues: list[str] | None = None,
    authors: list[str] | None = None,
    corpus_ids: list[CorpusId] | None = None,
    fields_of_study: list[str] | None = None,
    inserted_before: str | None = None,
) -> dict[str, str]:
    if corpus_ids and list(filter(None, corpus_ids)):
        # If you provide the corpus_ids filter, you can't provide other filters.
        return {"paperIds": ",".join([f"CorpusId:{cid}" for cid in corpus_ids])}
    filters = dict()
    if venues and list(filter(None, venues)):
        filters["venue"] = ",".join(venues)
    if authors and list(filter(None, authors)):
        # TODO - validate the api syntax for filtering by authors once available
        filters["authors"] = ",".join(authors)
    if time_range and time_range.non_empty():
        if time_range.start == time_range.end:
            filters["year"] = str(time_range.start)
        else:
            filters["year"] = (
                f"{str(time_range.start) if time_range.start else ''}-{str(time_range.end) if time_range.end else ''}"
            )
    if fields_of_study:
        filters["fieldsOfStudy"] = ",".join(fields_of_study)

    if inserted_before:
        # NOTE: currently if both are given, year will override inserted_before, so we cant build a range
        #   those differing to post filtering. but notice we assume agents are in charge of doing this.
        if "year" in filters:
            _ = filters.pop("year")
        filters["insertedBefore"] = inserted_before

    return filters


def get_paper_metadata(metadata: dict) -> dict:
    # they must have "metadata" and "corpus_id"
    url = f"https://api.semanticscholar.org/CorpusId:{metadata['corpusId']}"
    if metadata.get("pdf_hash"):
        # if pdfHash is set lets use it for full paper url
        url = f"https://www.semanticscholar.org/paper/{metadata['pdfHash']}"
    return {
        "corpus_id": metadata["corpusId"],
        "authors": ([Author(name=a) for a in metadata["authors"]] if metadata.get("authors") else []),
        "title": metadata["title"] if metadata.get("title") else "",
        "url": url,
        # TODO - add when available
        # "year": metadata["pubdate"]["year"] if metadata.get("pubdate") and metadata["pubdate"].get("year") else 0,
        # "venue": metadata["venue"]["name"] if metadata.get("venue") and metadata["venue"].get("name") else "",
        # "abstract": metadata["abstract"] if metadata.get("abstract") else "",
    }


def get_bounding_boxes(snippet: Any) -> list[BoundingBox]:
    ret = []
    # flattening as we dont have a notion of within snippet sentences (a snippet is a long sent)
    # TODO: change this now that we have a notion of within snippet sentences?
    for sent in snippet.get("sentences") if snippet.get("sentences") else []:
        for bb in sent.get("bbs") if sent.get("bbs") else []:
            try:
                ret.append(BoundingBox(**bb))
            except TypeError:
                logger.warning(f"couldn't extract bounding boxes information (continue with best effort): {bb}")
    return ret


def get_ref_mentions(snippet_annotations: SnippetAnnotations) -> list[dict[str, Any]]:
    if not snippet_annotations.get("refMentions"):
        return []
    return [
        {
            "matched_paper_corpus_id": cid["matchedPaperCorpusId"],
            "within_snippet_offset_start": cid["start"],
            "within_snippet_offset_end": cid["end"],
        }
        for cid in snippet_annotations["refMentions"]
        if cid.get("matchedPaperCorpusId") and cid.get("start") and cid.get("end")
    ]


def get_sentence_offsets(
    snippet_annotations: SnippetAnnotations, snippet_offset: dict[str, int]
) -> list[dict[str, dict[str, int]]]:
    ret = []
    for sent in snippet_annotations.get("sentences") if snippet_annotations.get("sentences") else []:
        sent_offsets = {
            "global_offset": {
                "start": snippet_offset["start"] + sent["start"],
                "end": snippet_offset["start"] + sent["end"],
            },
            "within_snippet_offset": sent,
        }
        ret.append(sent_offsets)
    return ret


class VespaRetriever(BaseRetriever):
    actual_vespa_version: str | None = None
    s2_client: AsyncSemanticScholar = Field()
    timeout: int = Field(default=10)

    def get_actual_vespa_version(self) -> str:
        if not self.actual_vespa_version:
            raise Exception("get_actual_vespa_version should be called only after retrieving docs")
        return self.actual_vespa_version

    def _init_docs(self, matches: dict) -> list[Document]:
        self.actual_vespa_version = matches["retrievalVersion"]

        docs = []
        for match in matches["data"]:
            s = match["snippet"]
            paper_metadata = get_paper_metadata(match["paper"])
            sentence_metadata = {
                "section_kind": s["snippetKind"],
                "section_title": (s["snippetKind"] if s["snippetKind"] in ["title", "abstract"] else s["section"]),
                "ref_mentions": get_ref_mentions(s.get("annotations", dict())),
                "sentence_offsets": get_sentence_offsets(s.get("annotations", dict()), s["snippetOffset"]),
                "bounding_boxes": None,  # TODO - add when available
            }
            offset_metadata = {
                "document_char_offsets": (
                    s["snippetOffset"]["start"],
                    s["snippetOffset"]["end"],
                )
            }
            docs.append(
                Document(
                    page_content=s["text"],
                    metadata={
                        "relevance_grade": "relevance",
                        "sentence": offset_metadata,
                        "metadata": {**paper_metadata, **sentence_metadata},
                        "dense_similarity": match["score"],
                    },
                )
            )
        docs = sorted(docs, key=lambda x: x.metadata["dense_similarity"], reverse=True)
        return docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        top_k: int = 10,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        authors: list[str] | None = None,
        corpus_ids: list[CorpusId] | None = None,
        fields_of_study: list[str] | None = None,
        inserted_before: str | None = None,
    ) -> list[Document]:
        raise NotImplementedError("Use the async version")

    @staticmethod
    def _validate_response(response: Response) -> dict:
        if response.status_code == 429:
            raise ExceedQuotaError()
        if 400 <= response.status_code < 500:
            text = response.text
            raise Exception(f"Request failed with status code {response.status_code}: {text}")
        if response.status_code == 504:
            raise TimeoutException("S2 API request timed out")
        if 500 <= response.status_code:
            text = response.text
            raise NetworkError(f"Request failed with status code {response.status_code}: {text}")
        json_response = response.json()
        if not isinstance(json_response, dict):
            raise ValueError(f"Expected a dictionary response, got {type(json_response)}")
        if "message" in json_response and json_response["message"] == "Endpoint request timed out":
            raise TimeoutException("S2 API request timed out")
        if "data" not in json_response:
            raise ValueError(f'Unexpected response from S2 snippet search API (missing "data" key): {response}')

        return json_response

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        top_k: int = 10,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        authors: list[str] | None = None,
        corpus_ids: list[CorpusId] | None = None,
        fields_of_study: list[str] | None = None,
        inserted_before: str | None = None,
    ) -> list[Document]:
        if authors:
            logger.warning(f"Filtering by authors is not supported yet. Got {authors=}")
        s2_client: AsyncSemanticScholar = self.s2_client
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{s2_client.api_url + s2_client.BASE_PATH_GRAPH}/snippet/search",
                params={
                    "query": query,
                    "limit": min(top_k, MAX_TOP_RESULTS),
                    **build_filters(
                        time_range=time_range,
                        venues=venues,
                        corpus_ids=corpus_ids,
                        fields_of_study=fields_of_study,
                        inserted_before=inserted_before,
                    ),  # TODO - add authors when feature is available
                },
                headers=s2_client.auth_header,
                timeout=httpx.Timeout(self.timeout),
            )

            matches = self._validate_response(response)
            return self._init_docs(matches)


class UniqueDocument(Document):
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, UniqueDocument):
            return False
        return self.page_content == other.page_content and self.metadata == other.metadata

    def __hash__(self) -> int:
        return hash(self.page_content) + hash(self.metadata.get("relevance_grade"))
