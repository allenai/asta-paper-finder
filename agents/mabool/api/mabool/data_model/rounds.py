from __future__ import annotations

import dataclasses
from dataclasses import field

from ai2i.dcollection import CorpusId
from pydantic import BaseModel, Field

from mabool.data_model.agent import AgentOperationMode


class RoundRequest(BaseModel):
    paper_description: str = Field(description="A description of the set of papers you're looking for.")
    operation_mode: AgentOperationMode | None = Field(
        default="infer", description='The operation mode of the tool: "infer" | "fast" | "diligent"'
    )
    inserted_before: str | None = Field(
        default=None,
        description="Excludes docs inserted on the provided date onward. Acceptable formats: YYYY-MM-DD, YYYY-MM, YYYY",
    )
    read_results_from_cache: bool | None = Field(default=False, description="Whether or not to use cached results.")


@dataclasses.dataclass
class RoundContext:
    message_id: str | None = field(default=None)
    thread_id: str | None = field(default=None)
    task_id: str | None = field(default=None)
    parent_step_id: str | None = field(default=None)
    caller_actor_id: str | None = field(default=None)
    cost_trace_id: str | None = field(default=None)
    channel_id: str | None = field(default=None)
    inserted_before: str | None = field(default=None)

    def fast_to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d
