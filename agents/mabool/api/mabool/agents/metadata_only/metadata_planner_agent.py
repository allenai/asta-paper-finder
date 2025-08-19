import ai2i.dcollection as dc
from ai2i.di import DI

from mabool.data_model.agent import AgentError, AgentInput, AgentOutput
from mabool.data_model.specifications import Specifications
from mabool.infra.operatives import (
    CompleteResponse,
    Operative,
    OperativeResponse,
    VoidResponse,
)
from mabool.utils import dc_deps

from .plan import plan


class MetadataPlannerInput(AgentInput):
    specification: Specifications


class MetadataPlannerAgent(Operative[MetadataPlannerInput, AgentOutput, None]):
    def register(self) -> None:
        pass

    @DI.managed
    async def handle_operation(
        self,
        state: None,
        inputs: MetadataPlannerInput,
        dcf: dc.DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory),
    ) -> tuple[None, OperativeResponse[AgentOutput]]:
        specifications = inputs.specification
        try:
            op = plan(specifications)
            op = op.build(dcf)
            docs = await op()
            return None, CompleteResponse(
                data=AgentOutput(
                    doc_collection=docs,
                    response_text="",
                )
            )
        except Exception as e:
            return None, VoidResponse(
                error=AgentError(
                    type="other",
                    message=str(e),
                )
            )
