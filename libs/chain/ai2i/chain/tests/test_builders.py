import uuid
from typing import TypedDict

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from pydantic import BaseModel

from ai2i.chain.builders import (
    define_chat_llm_call,
    define_model,
    define_prompt_llm_call,
    system_message,
    user_message,
)
from ai2i.chain.computation import ModelRunnableFactory
from ai2i.chain.tests.mocks import MockModelRunnable
from ai2i.chain.tests.test_computation import MockResponse


class SomeInput(TypedDict):
    input: str


class SomeResponse(BaseModel):
    value: str


@pytest.fixture
def output_value() -> str:
    return f"out_{uuid.uuid4()}"


@pytest.fixture
def input_value() -> str:
    return f"in_{uuid.uuid4()}"


@pytest.fixture
def custom_format() -> str:
    return f"fmt_{uuid.uuid4()}"


@pytest.fixture
def extra_param1() -> str:
    return f"exp1_{uuid.uuid4()}"


@pytest.fixture
def extra_param2() -> str:
    return f"exp2_{uuid.uuid4()}"


def test_model_define() -> None:
    model_mock = MockModelRunnable()
    model_mock.return_value = MockResponse(value="result").model_dump_json()
    model_factory: ModelRunnableFactory = lambda: model_mock

    runnable = define_model().build_runnable(model_factory)
    assert runnable == model_mock


async def test_simple_define_prompt_f_string(output_value: str, input_value: str) -> None:
    prompt_computation = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    model_mock = MockModelRunnable()
    model_mock.return_value = SomeResponse(value=output_value).model_dump_json()
    model_factory: ModelRunnableFactory = lambda: model_mock

    runnable = prompt_computation.build_runnable(model_factory)
    r = await runnable.ainvoke({"input": input_value})

    assert r == output_value
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, StringPromptValue)
    assert model_mock.last_input.text.startswith(input_value)  # format instructions should follow


async def test_simple_define_prompt_mustache(output_value: str, input_value: str) -> None:
    prompt_computation = define_prompt_llm_call(
        "{{input}}",
        format="mustache",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    model_mock = MockModelRunnable()
    model_mock.return_value = SomeResponse(value=output_value).model_dump_json()
    model_factory: ModelRunnableFactory = lambda: model_mock

    runnable = prompt_computation.build_runnable(model_factory)
    r = await runnable.ainvoke({"input": input_value})

    assert r == output_value
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, StringPromptValue)
    assert model_mock.last_input.text.startswith(input_value)  # format instructions should follow


async def test_custom_format_define_prompt_f_string(output_value: str, input_value: str, custom_format: str) -> None:
    prompt_computation = define_prompt_llm_call(
        "{input}", input_type=SomeInput, output_type=SomeResponse, custom_format_instructions=custom_format
    ).map(lambda s: s.value)

    model_mock = MockModelRunnable()
    model_mock.return_value = SomeResponse(value=output_value).model_dump_json()
    model_factory: ModelRunnableFactory = lambda: model_mock

    runnable = prompt_computation.build_runnable(model_factory)
    r = await runnable.ainvoke({"input": input_value})

    assert r == output_value
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, StringPromptValue)
    assert model_mock.last_input.text == f"{input_value}\n{custom_format}"


async def test_custom_format_define_prompt_mustache(output_value: str, input_value: str, custom_format: str) -> None:
    prompt_computation = define_prompt_llm_call(
        "{{input}}",
        format="mustache",
        input_type=SomeInput,
        output_type=SomeResponse,
        custom_format_instructions=custom_format,
    ).map(lambda s: s.value)

    model_mock = MockModelRunnable()
    model_mock.return_value = SomeResponse(value=output_value).model_dump_json()
    model_factory: ModelRunnableFactory = lambda: model_mock

    runnable = prompt_computation.build_runnable(model_factory)
    r = await runnable.ainvoke({"input": input_value})

    assert r == output_value
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, StringPromptValue)
    assert model_mock.last_input.text == f"{input_value}\n{custom_format}"


async def test_define_prompt_with_extra_params(
    output_value: str, input_value: str, custom_format: str, extra_param1: str, extra_param2: str
) -> None:
    prompt_computation = define_prompt_llm_call(
        "{{input}}-{{e1}}-{{e2}}",
        format="mustache",
        input_type=SomeInput,
        output_type=SomeResponse,
        get_extra_params=lambda: {"e1": extra_param1, "e2": extra_param2},
        custom_format_instructions=custom_format,
    ).map(lambda s: s.value)

    model_mock = MockModelRunnable()
    model_mock.return_value = SomeResponse(value=output_value).model_dump_json()
    model_factory: ModelRunnableFactory = lambda: model_mock

    runnable = prompt_computation.build_runnable(model_factory)
    r = await runnable.ainvoke({"input": input_value})

    assert r == output_value
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, StringPromptValue)
    assert model_mock.last_input.text == f"{input_value}-{extra_param1}-{extra_param2}\n{custom_format}"


async def test_define_prompt_with_response_metadata(output_value: str, input_value: str, extra_param1: str) -> None:
    prompt_computation = define_prompt_llm_call(
        "{input}", input_type=SomeInput, output_type=SomeResponse, include_response_metadata=True
    ).map(lambda t: (t[0].value, t[1]))

    model_mock = MockModelRunnable()
    model_mock.return_value = SomeResponse(value=output_value).model_dump_json()
    model_mock.return_metadata = {"mock_key": extra_param1}
    model_factory: ModelRunnableFactory = lambda: model_mock

    runnable = prompt_computation.build_runnable(model_factory)
    r, metadata = await runnable.ainvoke({"input": input_value})

    assert isinstance(metadata, dict)
    assert "mock_key" in metadata
    assert metadata["mock_key"] == extra_param1
    assert r == output_value
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, StringPromptValue)
    assert model_mock.last_input.text.startswith(input_value)  # format instructions should follow


async def test_define_chat_prompt(output_value: str, input_value: str, extra_param1: str) -> None:
    prompt_computation = define_chat_llm_call(
        [system_message("Hello"), user_message("{input}")],
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda t: t.value)

    model_mock = MockModelRunnable()
    model_mock.return_value = SomeResponse(value=output_value).model_dump_json()
    model_factory: ModelRunnableFactory = lambda: model_mock

    runnable = prompt_computation.build_runnable(model_factory)
    r = await runnable.ainvoke({"input": input_value})

    assert r == output_value
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, ChatPromptValue)
    assert model_mock.last_input.messages[0] == SystemMessage("Hello")
    assert model_mock.last_input.messages[1] == HumanMessage(input_value)


async def test_define_chat_prompt_with_response_metadata(
    output_value: str, input_value: str, extra_param1: str
) -> None:
    prompt_computation = define_chat_llm_call(
        [system_message("Hello"), user_message("{input}")],
        input_type=SomeInput,
        output_type=SomeResponse,
        include_response_metadata=True,
    ).map(lambda t: (t[0].value, t[1]))

    model_mock = MockModelRunnable()
    model_mock.return_value = SomeResponse(value=output_value).model_dump_json()
    model_mock.return_metadata = {"mock_key": extra_param1}
    model_factory: ModelRunnableFactory = lambda: model_mock

    runnable = prompt_computation.build_runnable(model_factory)
    r, metadata = await runnable.ainvoke({"input": input_value})

    assert isinstance(metadata, dict)
    assert "mock_key" in metadata
    assert metadata["mock_key"] == extra_param1

    assert r == output_value
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, ChatPromptValue)
    assert model_mock.last_input.messages[0] == SystemMessage("Hello")
    assert model_mock.last_input.messages[1] == HumanMessage(input_value)
