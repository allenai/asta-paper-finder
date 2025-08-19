import logging

from ai2i.config import config_value, ufv
from ai2i.dcollection import (
    DocumentCollection,
    ExtractedYearlyTimeRange,
)
from ai2i.di import DI

from mabool.agents.common.domain_utils import get_fields_of_study_filter_from_domains
from mabool.agents.common.utils import alog_args
from mabool.data_model.agent import (
    AgentError,
    AgentInput,
    AgentOutput,
    DomainsIdentified,
)
from mabool.data_model.config import cfg_schema
from mabool.data_model.ufs import uf
from mabool.infra.operatives import (
    CompleteResponse,
    Operative,
    OperativeResponse,
    VoidResponse,
)
from mabool.utils.dc import DC

logger = logging.getLogger(__name__)


class MetadataOnlySearchInput(AgentInput):
    time_range: ExtractedYearlyTimeRange | None = None
    venues: list[str] | None = None
    # NOTE: if venues exist, we ignore domains, as domains are often hallucinated, and venue already select for domain.
    domains: DomainsIdentified


class MetadataOnlySearchAgent(Operative[MetadataOnlySearchInput, AgentOutput, None]):
    def register(self) -> None: ...

    @DI.managed
    async def get_papers_by_metadata(
        self,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        domains: DomainsIdentified | None = None,
    ) -> DocumentCollection:
        assert venues or (time_range and not time_range.is_empty())

        # NOTE we get "computer science" by default ...
        fields_of_study = get_fields_of_study_filter_from_domains(domains) if domains else None

        search_results = await DC.from_s2_search(
            "",
            limit=config_value(cfg_schema.s2_api.total_papers_limit),
            time_range=time_range,
            venues=venues,
            fields_of_study=fields_of_study if not venues else None,
        )
        return search_results

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self,
        state: None,
        inputs: MetadataOnlySearchInput,
    ) -> tuple[None, OperativeResponse[AgentOutput]]:
        response_text = ""
        try:
            results = await self.get_papers_by_metadata(inputs.time_range, inputs.venues, inputs.domains)
            if not results or len(results.documents) == 0:
                response_text = ufv(uf.response_texts.metadata_agent.could_not_find_in_s2)
                if inputs.venues and len(inputs.venues) > 0:
                    response_text += ufv(uf.response_texts.metadata_agent.try_alternative)
            elif len(results.documents) == config_value(cfg_schema.s2_api.total_papers_limit):
                response_text = ufv(
                    uf.response_texts.metadata_agent.notice_limit,
                    limit=config_value(cfg_schema.s2_api.total_papers_limit),
                )
        except Exception as e:
            return None, VoidResponse(error=AgentError(type="other", message=str(e)))

        return (
            None,
            CompleteResponse(data=AgentOutput(response_text=response_text, doc_collection=results)),
        )
