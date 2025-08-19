from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypedDict,
    TypeVar,
    Unpack,
    overload,
)

import httpx
from google.genai import Client
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import get_config_list
from langchain_core.runnables.utils import gather_with_concurrency
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from tenacity import (
    RetryCallState,
    RetryError,
    after_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tenacity.asyncio.retry import RetryBaseT as AsyncRetryBaseT
from tenacity.retry import RetryBaseT
from tenacity.stop import StopBaseT
from tenacity.wait import WaitBaseT

from ai2i.chain.computation import ChainComputation, ModelRunnable
from ai2i.chain.gemini.async_genai import AsyncChatGoogleGenAI
from ai2i.chain.models import (
    DEFAULT_BATCH_MAX_CONCURRENCY,
    LLMModel,
    LLMModelParams,
    anthropic_model_params,
    google_model_params,
    openai_model_params,
)

default_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaboolTimeout:
    httpx_timeout: httpx.Timeout

    def total_seconds(self) -> float | None:
        total = 0.0
        if self.httpx_timeout.connect is not None:
            total += self.httpx_timeout.connect
        if self.httpx_timeout.read is not None:
            total += self.httpx_timeout.read
        if self.httpx_timeout.write is not None:
            total += self.httpx_timeout.write
        if self.httpx_timeout.pool is not None:
            total += self.httpx_timeout.pool

        if total == 0.0:
            return None
        return total


class Timeouts:
    tiny: ClassVar[MaboolTimeout] = MaboolTimeout(httpx.Timeout(10.0, read=5.0, write=3.0, connect=3.0))
    short: ClassVar[MaboolTimeout] = MaboolTimeout(httpx.Timeout(15.0, read=10.0, write=5.0, connect=15.0))
    medium: ClassVar[MaboolTimeout] = MaboolTimeout(httpx.Timeout(35.0, read=30.0, write=5.0, connect=15.0))
    long: ClassVar[MaboolTimeout] = MaboolTimeout(httpx.Timeout(60.0, read=30.0, write=45.0, connect=15.0))
    extra_long: ClassVar[MaboolTimeout] = MaboolTimeout(httpx.Timeout(120.0, read=60.0, write=90.0, connect=30.0))


class TenacityRetrySettings(TypedDict, total=False):
    sleep: Callable[[int | float], Awaitable[None] | None]
    stop: StopBaseT
    wait: WaitBaseT
    retry: RetryBaseT | AsyncRetryBaseT
    before: Callable[[RetryCallState], Awaitable[None] | None]
    after: Callable[[RetryCallState], Awaitable[None] | None]
    before_sleep: Callable[[RetryCallState], Awaitable[None] | None] | None
    reraise: bool
    retry_error_cls: type[RetryError]
    retry_error_callback: Callable[[RetryCallState], Awaitable[Any] | Any] | None


def default_retry_settings(logger: logging.Logger | None = None) -> TenacityRetrySettings:
    return {
        # mabool defaults
        "wait": wait_random_exponential(multiplier=0.5, max=10),
        "stop": stop_after_attempt(5),
        "after": after_log(logger if logger is not None else default_logger, logging.DEBUG),
    }


class BatchExecutionContextBase(TypedDict, total=False):
    max_concurrency: int


# We separate return_exceptions because this property influances the return type from a batch call
# so it will need to be treated differently from the other props in function signatures
class BatchExecutionContext(BatchExecutionContextBase, total=False):
    return_exceptions: bool


def _default_batch_execution_context() -> BatchExecutionContext:
    return {
        "max_concurrency": DEFAULT_BATCH_MAX_CONCURRENCY,
        "return_exceptions": False,
    }


IN = TypeVar("IN", contravariant=True)
OUT = TypeVar("OUT", covariant=True)


class LangChainModelFactory(Protocol):
    def __call__(self, model: LLMModel, timeout: MaboolTimeout, api_key: SecretStr | None = None) -> BaseChatModel: ...


def _openai_chat_factory(model: LLMModel, timeout: MaboolTimeout, api_key: SecretStr | None = None) -> BaseChatModel:
    params = openai_model_params(model.params)

    return ChatOpenAI(
        api_key=api_key,
        model=model.name,
        model_kwargs={"response_format": {"type": "json_object"}},
        timeout=timeout.httpx_timeout,
        **params,
    )


def _anthropic_chat_factory(model: LLMModel, timeout: MaboolTimeout, api_key: SecretStr | None = None) -> BaseChatModel:
    secret_api_key = SecretStr("") if api_key is None else api_key
    params = anthropic_model_params(model.params)

    return ChatAnthropic(api_key=secret_api_key, model_name=model.name, timeout=timeout.total_seconds(), **params)


def _google_chat_factory_with_async(
    model: LLMModel, timeout: MaboolTimeout, api_key: SecretStr | None = None
) -> BaseChatModel:
    params = google_model_params(model.params)

    from google.genai.types import HttpOptions

    timeout_seconds = timeout.total_seconds()
    secret_api_key = str(api_key.get_secret_value()) if api_key else None
    return AsyncChatGoogleGenAI(
        model_name=model.name,
        client=Client(
            api_key=secret_api_key,
            http_options=HttpOptions(timeout=int(1000 * timeout_seconds) if timeout_seconds is not None else None),
        ),
        model_kwargs={
            "generation_config": {
                "response_mime_type": "application/json",
            },
            **params,
        },
    )


class JsonChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await super().ainvoke(
            *args,
            generation_config={
                "response_mime_type": "application/json",
            },
            **kwargs,
        )


def _google_chat_factory(model: LLMModel, timeout: MaboolTimeout, api_key: SecretStr | None = None) -> BaseChatModel:
    secret_api_key = None if api_key is None else api_key
    params = google_model_params(model.params)

    return JsonChatGoogleGenerativeAI(
        api_key=secret_api_key, model=model.name, timeout=timeout.total_seconds(), **params
    )


TrueLiteral = Literal[True]
FalseLiteral = Literal[False]


@dataclass(frozen=True)
class Execution(Generic[IN, OUT]):
    runnable: Runnable[IN, OUT]

    async def once(self, input: IN) -> OUT:
        return await self.runnable.ainvoke(input)

    # The following overload changes the return type based on the 'return_exception' parameter. If it's set to 'True'
    # The resulting list will contain elements of 'OUT | Exception', if it's False it will only contain 'OUT' elements
    @overload
    async def many(
        self, inputs: list[IN], *, return_exceptions: TrueLiteral, **batch_ec: Unpack[BatchExecutionContextBase]
    ) -> list[OUT | Exception]:
        pass

    @overload
    async def many(
        self,
        inputs: list[IN],
        *,
        return_exceptions: FalseLiteral = False,
        **batch_ec: Unpack[BatchExecutionContextBase],
    ) -> list[OUT]:
        pass

    async def many(
        self, inputs: list[IN], return_exceptions: bool = False, **batch_ec: Unpack[BatchExecutionContextBase]
    ) -> list[OUT | Exception] | list[OUT]:
        resolved_batch_ec: BatchExecutionContext = {
            **_default_batch_execution_context(),
            **batch_ec,
            "return_exceptions": return_exceptions,
        }
        return await self.runnable.abatch(
            inputs,
            config={"max_concurrency": resolved_batch_ec.get("max_concurrency")},
            return_exceptions=resolved_batch_ec["return_exceptions"],
        )


class RetryWithTenacity(Runnable[IN, OUT]):
    _decorated: Runnable[IN, OUT]
    _retry_settings: TenacityRetrySettings

    def __init__(self, decorated: Runnable[IN, OUT], retry_settings: TenacityRetrySettings) -> None:
        self._decorated = decorated
        self._retry_settings = retry_settings
        super().__init__()

    async def ainvoke(self, input: IN, config: RunnableConfig | None = None, **kwargs: Any) -> OUT:
        @retry(**self._retry_settings)
        async def invoke_with_retry() -> OUT:
            return await self._decorated.ainvoke(input, config, **kwargs)

        return await invoke_with_retry()

    def invoke(self, input: IN, config: RunnableConfig | None = None, **kwargs: Any) -> OUT:
        raise NotImplementedError(f"No support for blocking calls in {self.__class__.__name__}")

    async def abatch(
        self,
        inputs: list[IN],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[OUT]:
        # NOTE:this is the implementation from Runnable, in case of a batch we want
        #      retries to be on individual calls and not on the bach itself
        if not inputs:
            return []

        configs = get_config_list(config, len(inputs))

        async def ainvoke(input: IN, config: RunnableConfig) -> OUT | Exception:
            if return_exceptions:
                try:
                    return await self.ainvoke(input, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return await self.ainvoke(input, config, **kwargs)

        coros = map(ainvoke, inputs, configs)
        return await gather_with_concurrency(configs[0].get("max_concurrency"), *coros)

    def batch(
        self,
        inputs: list[IN],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[OUT]:
        raise NotImplementedError(f"No support for blocking calls in {self.__class__.__name__}")


@dataclass(frozen=True)
class LLMEndpoint:
    default_retry_settings: TenacityRetrySettings
    default_timeout: MaboolTimeout
    default_model: LLMModel
    model_factory: LangChainModelFactory
    api_key: SecretStr | None = None

    def timeout(self, timeout: MaboolTimeout) -> LLMEndpoint:
        return LLMEndpoint(self.default_retry_settings, timeout, self.default_model, self.model_factory, self.api_key)

    def model(self, model: LLMModel) -> LLMEndpoint:
        return LLMEndpoint(self.default_retry_settings, self.default_timeout, model, self.model_factory, self.api_key)

    def model_params(self, **params: Unpack[LLMModelParams]) -> LLMEndpoint:
        return LLMEndpoint(
            self.default_retry_settings,
            self.default_timeout,
            self.default_model.override(**params),
            self.model_factory,
            self.api_key,
        )

    def retry_settings(self, **kwargs: Unpack[TenacityRetrySettings]) -> LLMEndpoint:
        return LLMEndpoint(
            {**self.default_retry_settings, **kwargs},
            self.default_timeout,
            self.default_model,
            self.model_factory,
            self.api_key,
        )

    def with_api_key(self, api_key: SecretStr) -> LLMEndpoint:
        return LLMEndpoint(
            self.default_retry_settings, self.default_timeout, self.default_model, self.model_factory, api_key
        )

    def execute(self, computation: ChainComputation[IN, OUT]) -> Execution[IN, OUT]:
        def model_factory() -> ModelRunnable:
            model = self.model_factory(self.default_model, self.default_timeout, self.api_key)
            # decorate with retries
            return RetryWithTenacity(model, self.default_retry_settings)

        runnable = computation.build_runnable(model_factory)
        return Execution(runnable)


def define_llm_endpoint(
    *,
    default_timeout: MaboolTimeout | None = None,
    default_model: LLMModel | None = None,
    logger: logging.Logger | None = None,
    api_key: SecretStr | None = None,
    **retry_settings: Unpack[TenacityRetrySettings],
) -> LLMEndpoint:
    def default_endpoint_from_model(model: LLMModel) -> LLMEndpoint:
        match model.family:
            case "openai":
                factory = _openai_chat_factory
            case "anthropic":
                factory = _anthropic_chat_factory
            case "google":
                factory = _google_chat_factory_with_async
            case family:
                raise ValueError(
                    f"Invalid family name: {family}, supported families are `openai`, `anthropic`, and `google`."
                )

        return LLMEndpoint(
            {**default_retry_settings(logger), **retry_settings},
            default_timeout if default_timeout is not None else Timeouts.short,
            model,
            factory,
            api_key,
        )

    if default_model is None:
        return LLMEndpoint(
            {**default_retry_settings(logger), **retry_settings},
            default_timeout if default_timeout is not None else Timeouts.short,
            LLMModel.gpt4o(),
            _openai_chat_factory,
            api_key,
        )

    return default_endpoint_from_model(default_model)
