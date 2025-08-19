import os
from abc import ABC, abstractmethod
from typing import Literal, Optional, Sequence

import cohere
from ai2i.common.utils.batch import batch_process
from ai2i.config import ConfigValue, config_value, configurable
from pydantic import BaseModel, Field

from mabool.data_model.config import cfg_schema


@configurable
def get_cohere_client(
    cohere_api_key: str | None = ConfigValue(cfg_schema.cohere_api_key, default=None),
) -> cohere.AsyncClient:
    if cohere_api_key is not None:
        api_key = cohere_api_key
    elif os.environ.get("COHERE_API_KEY"):
        api_key = os.environ["COHERE_API_KEY"]
    elif os.path.exists("cohere_api_key.txt"):
        api_key = open("cohere_api_key.txt", "r").read().strip()
    else:
        raise ValueError("Cohere API key not found")
    return cohere.AsyncClient(api_key, timeout=60)


class RerankScoreResult(BaseModel):
    corpus_id: str
    score: float


class RerankScoreOutput(BaseModel):
    results: list[RerankScoreResult]
    method: Literal["cohere"]
    rerank_model_name: Optional[str] = None


class RerankScoreDocInput(BaseModel):
    corpus_id: str
    text: str


class RerankScoreInput(BaseModel):
    query: str
    docs: list[RerankScoreDocInput]


class RerankScorer(ABC):
    @abstractmethod
    async def rerank(self, rerank_input: RerankScoreInput) -> RerankScoreOutput: ...


type CohereModels = Literal["rerank-english-v2.0", "rerank-english-v3.0"]


class CohereRerankScorer(BaseModel, RerankScorer):
    rerank_model_name: CohereModels = "rerank-english-v3.0"
    client: cohere.AsyncClient = Field(default_factory=get_cohere_client)

    class Config:
        arbitrary_types_allowed = True

    async def rerank(self, rerank_input: RerankScoreInput) -> RerankScoreOutput:
        async def rerank_batch(documents: Sequence[RerankScoreDocInput]) -> Sequence[RerankScoreResult]:
            doc_texts_with_id = [(doc.corpus_id, doc.text) for doc in documents]
            corpus_ids, doc_texts = zip(*doc_texts_with_id)

            cohere_results = await self.client.rerank(
                query=rerank_input.query,
                documents=doc_texts,
                model=self.rerank_model_name,
                return_documents=True,
                request_options={"max_retries": 3},
            )

            return [
                RerankScoreResult(corpus_id=corpus_ids[result.index], score=result.relevance_score)
                for result in cohere_results.results
            ]

        results: Sequence[RerankScoreResult] = await batch_process(
            rerank_input.docs,
            batch_size=500,
            process_func=rerank_batch,
            max_concurrency=2,
            force_deterministic=config_value(cfg_schema.force_deterministic),
        )

        return RerankScoreOutput(results=list(results), method="cohere", rerank_model_name=self.rerank_model_name)
