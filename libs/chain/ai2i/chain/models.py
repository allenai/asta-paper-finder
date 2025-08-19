from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict, Unpack

# Default LLM parameters
DEFAULT_TEMPERATURE = 1
DEFAULT_N = 1
DEFAULT_BATCH_MAX_CONCURRENCY = 5

# Default model names
GPT4_DEFAULT_MODEL = "gpt-4"
GPT4O_DEFAULT_MODEL = "gpt-4o"
GPT4TURBO_DEFAULT_MODEL = "gpt-4-turbo"
CLAUDE35SONNET_DEFAULT_MODEL = "claude-3-5-sonnet-20240620"
GEMINI2FLASH_DEFAULT_MODEL = "gemini-2.0-flash"


class LLMModelParams(TypedDict, total=False):
    temperature: float
    top_p: int
    n: int
    max_tokens: int


def openai_model_params(params: LLMModelParams) -> dict[str, Any]:
    return {
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "n": params.get("n"),
        "max_tokens": params.get("max_tokens"),
        "max_retries": 0,
    }


def anthropic_model_params(params: LLMModelParams) -> dict[str, Any]:
    return {
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "max_tokens_to_sample": params.get("max_tokens", 1024),
        "max_retries": 0,
    }


def google_model_params(params: LLMModelParams) -> dict[str, Any]:
    return {
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "candidate_count": params.get("n"),
        "max_output_tokens": params.get("max_tokens"),
    }


ModelName = Literal[
    "openai:gpt4-default",
    "openai:gpt4o-default",
    "openai:gpt4turbo-default",
    "anthropic:claude35sonnet-default",
    "google:gemini2flash-default",
]


@dataclass(frozen=True)
class LLMModel:
    name: str
    family: str
    params: LLMModelParams

    @staticmethod
    def from_name(name: str, **params: Unpack[LLMModelParams]) -> LLMModel:
        match name:
            case "openai:gpt4-default":
                return LLMModel.gpt4(**params)
            case "openai:gpt4o-default":
                return LLMModel.gpt4o(**params)
            case "openai:gpt4turbo-default":
                return LLMModel.gpt4turbo(**params)
            case "anthropic:claude35sonnet-default":
                return LLMModel.claude35sonnet(**params)
            case "google:gemini2flash-default":
                return LLMModel.gemini2flash(**params)
            case model_name:
                if ":" not in model_name:
                    raise ValueError("LLM model name must be in the format `family:model`.")
                family, model = model_name.split(":", maxsplit=1)
                return LLMModel(name=model, family=family, params={**LLMModel._get_default_llm_params(), **params})

    @staticmethod
    def _get_default_llm_params() -> LLMModelParams:
        return {
            "temperature": DEFAULT_TEMPERATURE,
            "n": DEFAULT_N,
        }

    @staticmethod
    def gpt4o(**params: Unpack[LLMModelParams]) -> LLMModel:
        return LLMModel(
            name=GPT4O_DEFAULT_MODEL,
            family="openai",
            params={**LLMModel._get_default_llm_params(), **params},
        )

    @staticmethod
    def gpt4turbo(**params: Unpack[LLMModelParams]) -> LLMModel:
        return LLMModel(
            name=GPT4TURBO_DEFAULT_MODEL,
            family="openai",
            params={**LLMModel._get_default_llm_params(), **params},
        )

    @staticmethod
    def gpt4(**params: Unpack[LLMModelParams]) -> LLMModel:
        return LLMModel(
            name=GPT4_DEFAULT_MODEL,
            family="openai",
            params={**LLMModel._get_default_llm_params(), **params},
        )

    @staticmethod
    def claude35sonnet(**params: Unpack[LLMModelParams]) -> LLMModel:
        return LLMModel(
            name=CLAUDE35SONNET_DEFAULT_MODEL,
            family="anthropic",
            params={**LLMModel._get_default_llm_params(), **params},
        )

    @staticmethod
    def gemini2flash(**params: Unpack[LLMModelParams]) -> LLMModel:
        return LLMModel(
            name=GEMINI2FLASH_DEFAULT_MODEL,
            family="google",
            params={**LLMModel._get_default_llm_params(), **params},
        )

    def override(self, **updated_params: Unpack[LLMModelParams]) -> LLMModel:
        return LLMModel(self.name, self.family, {**self.params, **updated_params})
