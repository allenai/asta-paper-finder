import logging

from ai2i.chain import LLMEndpoint, LLMModel, Timeouts, define_llm_endpoint
from ai2i.config import config_value
from ai2i.dcollection import (
    AssignedField,
    DocumentCollection,
    ExtractedYearlyTimeRange,
    PaperFinderDocument,
    get_by_title_origin_query,
)
from ai2i.di import DI

from mabool.agents.common.common import (
    AgentState,
    filter_by_author,
)
from mabool.agents.common.utils import alog_args
from mabool.agents.specific_paper_by_title.specific_paper_by_title_prompts import title_extraction
from mabool.data_model.agent import (
    AgentError,
    AgentInput,
    AgentOutput,
)
from mabool.data_model.config import cfg_schema
from mabool.infra.operatives import (
    CompleteResponse,
    Operative,
    OperativeResponse,
    VoidResponse,
)
from mabool.utils.dc import DC
from mabool.utils.llm_utils import get_api_key_for_model

logger = logging.getLogger(__name__)


class SpecificPaperByTitleInput(AgentInput):
    matched_title: str
    matched_corpus_ids: list[str]
    time_range: ExtractedYearlyTimeRange | None = None
    venues: list[str] | None = None


type SpecificPaperByTitleState = AgentState
type SpecificPaperByTitleOutput = AgentOutput


def get_default_endpoint() -> LLMEndpoint:
    llm_model = LLMModel.from_name(
        config_value(cfg_schema.specific_paper_by_title_agent.llm_model_name), temperature=0.0
    )
    return define_llm_endpoint(
        default_timeout=Timeouts.medium,
        default_model=llm_model,
        logger=logger,
        api_key=get_api_key_for_model(llm_model),
    )


def _titles_match(title1: str | None, title2: str | None) -> bool:
    def normalize_title(t: str) -> str:
        t = "".join(filter(str.isalnum, t.lower()))
        return "".join([w for w in t.split() if len(w) > 3])

    if title1 is None or title2 is None:
        return False
    matching = normalize_title(title1) == normalize_title(title2)
    return matching


@DI.managed
async def get_specific_paper_by_title(
    user_input: str,
    time_range: ExtractedYearlyTimeRange | None = None,
    venues: list[str] | None = None,
    authors: list[str] | None = None,
) -> tuple[DocumentCollection, str]:
    extracted_title = None
    try:
        extracted_title = await get_default_endpoint().execute(title_extraction).once(user_input)
    except Exception as e:
        logger.warning(f"Failed extracting title from query, fallback to entire query's content: {e}")

    if not extracted_title:
        extracted_title = user_input

    search_results = await DC.from_s2_by_title(extracted_title, time_range, venues)
    search_results = search_results.filter(lambda doc: _titles_match(doc.title, extracted_title))

    if authors:
        search_results = search_results.filter(lambda doc: filter_by_author(authors, doc))

    return search_results, extracted_title


class SpecificPaperByTitleAgent(
    Operative[SpecificPaperByTitleInput, SpecificPaperByTitleOutput, SpecificPaperByTitleState]
):
    def register(self) -> None: ...

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self, state: SpecificPaperByTitleState | None, inputs: SpecificPaperByTitleInput
    ) -> tuple[SpecificPaperByTitleState | None, OperativeResponse[SpecificPaperByTitleOutput]]:
        try:
            search_results = DC.from_docs(
                [
                    PaperFinderDocument(
                        corpus_id=corpus_id,
                        origins=[get_by_title_origin_query(inputs.matched_title, inputs.time_range, inputs.venues)],
                    )
                    for corpus_id in inputs.matched_corpus_ids
                ]
            )
            if len(search_results) == 0:
                return (
                    state,
                    CompleteResponse(
                        data=AgentOutput(
                            response_text="",
                            doc_collection=search_results,
                        )
                    ),
                )

            search_results = await search_results.with_fields(
                [
                    AssignedField[float](
                        field_name="final_specific_paper_by_title_score",
                        assigned_values=[1.0] * len(search_results),
                    ),
                ]
            )

        except Exception as e:
            return None, VoidResponse(error=AgentError(type="other", message=str(e)))

        return (
            state,
            CompleteResponse(data=AgentOutput(response_text="", doc_collection=search_results)),
        )
