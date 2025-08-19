from ai2i.chain import LLMModel
from ai2i.config import config_value
from pydantic import SecretStr

from mabool.data_model.config import cfg_schema


def get_api_key_for_model(model: LLMModel) -> SecretStr | None:
    api_key: str | None = None

    match model.family:
        case "openai":
            api_key = config_value(cfg_schema.openai_api_key, default=None)
        case "anthropic":
            api_key = config_value(cfg_schema.anthropic_api_key, default=None)
        case "google":
            api_key = config_value(cfg_schema.google_api_key, default=None)
        case _:
            pass

    return SecretStr(api_key) if api_key else None
