from __future__ import annotations

from ai2i.di import DI, create_module
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

from mabool.dal import dal_deps
from mabool.data_model.config import cfg_schema
from mabool.external_api import external_api_deps
from mabool.utils import context_deps, dc_deps, tracing_deps

services_module = create_module(
    "Services",
    extends=[
        dal_deps.dal_module,
        external_api_deps.external_api_module,
        context_deps.context_module,
        dc_deps.dc_module,
        tracing_deps.tracing_module,
    ],
)


@services_module.global_init()
async def _setup_globals(
    use_cache: bool = DI.config(cfg_schema.enable_llm_cache),
) -> None:
    if use_cache:
        set_llm_cache(InMemoryCache())

    return None
