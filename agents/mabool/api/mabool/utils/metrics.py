from collections import defaultdict
from datetime import datetime

from ai2i.common.utils.time import get_utc_time
from pydantic import BaseModel, Field

from mabool.data_model.agent import AggregatedMetrics


class Metrics(BaseModel):
    corpus_ids_relevance_judged: defaultdict[str, int] = Field(default_factory=lambda: defaultdict(int))
    start_time: datetime = Field(default_factory=get_utc_time)
    relevance_judgement_failures: int = 0

    def add_relevance_judged(self, corpus_id: str) -> None:
        self.corpus_ids_relevance_judged[corpus_id] += 1

    def aggregate(self) -> AggregatedMetrics:
        return AggregatedMetrics(
            start_time=self.start_time,
            duration=(get_utc_time() - self.start_time).total_seconds(),
            num_docs_judged=len(self.corpus_ids_relevance_judged.keys()),
            num_relevance_judgement_calls=sum(self.corpus_ids_relevance_judged.values()),
            relevance_judgement_failures=self.relevance_judgement_failures,
        )
