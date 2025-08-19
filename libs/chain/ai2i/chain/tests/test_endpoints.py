import uuid
from typing import TypedDict, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import StringPromptValue
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, SecretStr
from tenacity import RetryError, stop_after_attempt

from ai2i.chain.builders import define_prompt_llm_call
from ai2i.chain.computation import ChainComputation
from ai2i.chain.endpoints import (
    LLMEndpoint,
    MaboolTimeout,
    Timeouts,
    default_retry_settings,
    define_llm_endpoint,
)
from ai2i.chain.models import LLMModel
from ai2i.chain.tests.mocks import MockModelRunnable


class SomeInput(TypedDict):
    input: str


class SomeResponse(BaseModel):
    value: str


@pytest.fixture
def input_value() -> str:
    return f"in1_{uuid.uuid4()}"


@pytest.fixture
def input_value2() -> str:
    return f"in2_{uuid.uuid4()}"


@pytest.fixture
def output_value() -> str:
    return f"out1_{uuid.uuid4()}"


@pytest.fixture
def output_value2() -> str:
    return f"out2_{uuid.uuid4()}"


@pytest.fixture
def api_key() -> SecretStr:
    return SecretStr("API_KEY")


def test_endpoint_overrides() -> None:
    endpoint = define_llm_endpoint()

    endpoint_with_overrides = (
        endpoint.timeout(Timeouts.extra_long)
        .retry_settings(reraise=True)
        .model(LLMModel.gpt4turbo(temperature=0.2))
        .model_params(temperature=0.64)
    )

    assert endpoint_with_overrides.default_model.name == LLMModel.gpt4turbo().name
    assert endpoint_with_overrides.default_model.params.get("temperature") == 0.64
    assert endpoint_with_overrides.default_retry_settings.get("reraise") is True
    assert endpoint_with_overrides.default_timeout == Timeouts.extra_long


async def test_run(input_value: str, input_value2: str, api_key: SecretStr) -> None:
    call = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    def _pass_through_model_factory(
        model: LLMModel, timeout: MaboolTimeout, api_key: SecretStr | None = None
    ) -> BaseChatModel:
        def _internal(input: StringPromptValue) -> BaseMessage:
            return BaseMessage(content=SomeResponse(value=input.text).model_dump_json(), type="test")

        return cast(BaseChatModel, RunnableLambda(_internal))

    endpoint = LLMEndpoint(
        default_retry_settings=default_retry_settings(),
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt4(),
        model_factory=_pass_through_model_factory,
        api_key=api_key,
    )

    r = await endpoint.execute(call).once({"input": input_value})

    assert r.startswith(input_value)  # followed by injected format instructions

    rb = await endpoint.execute(call).many([{"input": input_value}, {"input": input_value2}])

    assert rb[0].startswith(input_value)  # followed by injected format instructions
    assert rb[1].startswith(input_value2)  # followed by injected format instructions


async def test_retries_fail(input_value: str, output_value: str, api_key: SecretStr) -> None:
    error_throws = 2

    call = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    model_mock = MockModelRunnable()
    model_mock.return_value = SomeResponse(value=output_value).model_dump_json()
    model_mock.throw_error_n_times = error_throws

    endpoint = LLMEndpoint(
        default_retry_settings={**default_retry_settings(), "stop": stop_after_attempt(error_throws)},
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt4(),
        model_factory=lambda model, timeout, api_key=None: cast(BaseChatModel, model_mock),
        api_key=api_key,
    )

    with pytest.raises(RetryError):
        await endpoint.execute(call).once({"input": input_value})


async def test_retries_success(input_value: str, output_value: str, api_key: SecretStr) -> None:
    error_throws = 2

    call = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    model_mock = MockModelRunnable()
    model_mock.return_value = SomeResponse(value=output_value).model_dump_json()
    model_mock.throw_error_n_times = error_throws

    endpoint = LLMEndpoint(
        default_retry_settings={**default_retry_settings(), "stop": stop_after_attempt(error_throws + 1)},
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt4(),
        model_factory=lambda model, timeout, api_key=None: cast(BaseChatModel, model_mock),
        api_key=api_key,
    )

    r = await endpoint.execute(call).once({"input": input_value})
    assert r == output_value


async def test_retries_multi_call_one_fully_fail(
    input_value: str, output_value: str, output_value2: str, api_key: SecretStr
) -> None:
    error_throws = 2

    call1 = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    call2 = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    call = ChainComputation.map_n(lambda x, y: (x, y), call1, call2)

    model_mock1 = MockModelRunnable()
    model_mock1.return_value = SomeResponse(value=output_value).model_dump_json()

    model_mock2 = MockModelRunnable()
    model_mock2.return_value = SomeResponse(value=output_value2).model_dump_json()
    model_mock2.throw_error_n_times = error_throws

    # NOTE: Each mock appears twice: once for the base model, once for the output-fixing model
    model_mocks = [model_mock1, model_mock1, model_mock2, model_mock2]

    def model_factory(model: LLMModel, timeout: MaboolTimeout, api_key: SecretStr | None = None) -> BaseChatModel:
        chosen_model = model_mocks.pop(0)
        return cast(BaseChatModel, chosen_model)

    endpoint = LLMEndpoint(
        default_retry_settings={**default_retry_settings(), "stop": stop_after_attempt(error_throws)},
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt4(),
        model_factory=model_factory,
        api_key=api_key,
    )

    with pytest.raises(RetryError):
        await endpoint.execute(call).once({"input": input_value})


async def test_retries_multi_call_both_fail_recover(
    input_value: str, output_value: str, output_value2: str, api_key: SecretStr
) -> None:
    error_throws = 2

    call1 = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    call2 = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    call = ChainComputation.map_n(lambda x, y: (x, y), call1, call2)

    model_mock1 = MockModelRunnable()
    model_mock1.return_value = SomeResponse(value=output_value).model_dump_json()
    model_mock1.throw_error_n_times = error_throws

    model_mock2 = MockModelRunnable()
    model_mock2.return_value = SomeResponse(value=output_value2).model_dump_json()
    model_mock2.throw_error_n_times = error_throws

    # NOTE: Each mock appears twice: once for the base model, once for the output-fixing model
    model_mocks = [model_mock1, model_mock1, model_mock2, model_mock2]

    def model_factory(model: LLMModel, timeout: MaboolTimeout, api_key: SecretStr | None = None) -> BaseChatModel:
        chosen_model = model_mocks.pop(0)
        return cast(BaseChatModel, chosen_model)

    endpoint = LLMEndpoint(
        default_retry_settings={**default_retry_settings(), "stop": stop_after_attempt(error_throws + 1)},
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt4(),
        model_factory=model_factory,
        api_key=api_key,
    )

    r = await endpoint.execute(call).once({"input": input_value})
    assert r == (output_value, output_value2)
