from ai2i.di import DI, create_module

from mabool.data_model.config import cfg_schema
from mabool.external_api.rerank.cohere import CohereModels, CohereRerankScorer, RerankScorer, get_cohere_client

external_api_module = create_module("ExternalAPI")


@external_api_module.provides(scope="singleton")
async def rerank_scorer(
    cohere_api_key: str | None = DI.config(cfg_schema.cohere_api_key, default=None),
    rerank_model_name: CohereModels = DI.config(cfg_schema.cohere.rerank_model_name),
) -> RerankScorer:
    cohere_client = get_cohere_client(cohere_api_key)
    return CohereRerankScorer(client=cohere_client, rerank_model_name=rerank_model_name)
