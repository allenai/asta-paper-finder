from ai2i.config import ConfigValue, configurable
from ai2i.dcollection import DocumentCollection, DocumentCollectionFactory
from ai2i.di import DI, create_module

from mabool.data_model.config import cfg_schema

dc_module = create_module("DocumentCollection")


@dc_module.provides(scope="singleton")
async def round_doc_collection_factory(
    s2_api_key: str = DI.config(cfg_schema.s2_api_key),
    s2_api_timeout: int = DI.config(cfg_schema.s2_api.timeout),
    cache_ttl: int = DI.config(cfg_schema.cache.ttl),
    cache_is_enabled: bool = DI.config(cfg_schema.cache.enabled),
    force_deterministic: bool = DI.config(cfg_schema.force_deterministic),
) -> DocumentCollectionFactory:
    dc_factory = DocumentCollectionFactory(
        s2_api_key=s2_api_key,
        s2_api_timeout=s2_api_timeout,
        cache_ttl=cache_ttl,
        cache_is_enabled=cache_is_enabled,
        force_deterministic=force_deterministic,
    )
    return dc_factory


@configurable
def detached_doc_collection_factory(
    s2_api_key: str = ConfigValue(cfg_schema.s2_api_key),
    s2_api_timeout: int = ConfigValue(cfg_schema.s2_api.timeout),
    cache_ttl: int = ConfigValue(cfg_schema.cache.ttl),
    cache_is_enabled: bool = ConfigValue(cfg_schema.cache.enabled),
    force_deterministic: bool = ConfigValue(cfg_schema.force_deterministic),
) -> DocumentCollectionFactory:
    dc_factory = DocumentCollectionFactory(
        s2_api_key=s2_api_key,
        s2_api_timeout=s2_api_timeout,
        cache_ttl=cache_ttl,
        cache_is_enabled=cache_is_enabled,
        force_deterministic=force_deterministic,
    )
    return dc_factory


@dc_module.provides(scope="singleton")
async def empty_doc_collection(
    dmf: DocumentCollectionFactory = DI.requires(round_doc_collection_factory),
) -> DocumentCollection:
    return dmf.empty()
