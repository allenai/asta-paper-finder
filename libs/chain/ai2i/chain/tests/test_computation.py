import random
import uuid
from typing import cast

import pytest
from langchain_core.messages.base import BaseMessage
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.runnables import RunnableBinding, RunnableLambda
from pydantic.main import BaseModel

from ai2i.chain.builders import define_model
from ai2i.chain.computation import ChainComputation, ModelRunnableFactory
from ai2i.chain.tests.mocks import MockModelRunnable


class MockResponse(BaseModel):
    value: str


def _to_int(s: BaseMessage | str) -> int:
    if isinstance(s, str):
        return int(s)
    if isinstance(s, BaseMessage):
        content = cast(str, s.content)
        return int(content)


def _to_float(s: BaseMessage | str) -> float:
    if isinstance(s, str):
        return float(s)
    if isinstance(s, BaseMessage):
        content = cast(str, s.content)
        return int(content)


def _plus_one(i: int) -> int:
    return i + 1


def _plus_two(i: int) -> int:
    return i + 2


def _to_prompt_value(s: str) -> PromptValue:
    return StringPromptValue(text=s)


@pytest.fixture
def random_string() -> str:
    return f"str_{uuid.uuid4()}"


@pytest.fixture
def random_int() -> int:
    return random.randint(0, 10000)


@pytest.fixture
def random_float() -> float:
    return random.random()


async def test_lift_and_basic_sequencing(random_string: str, random_int: int) -> None:
    model_mock = MockModelRunnable()
    model_mock.return_value = str(random_int)
    model_factory: ModelRunnableFactory = lambda: model_mock

    comp = define_model() | ChainComputation.lift(_to_int) | ChainComputation.lift(RunnableLambda(_plus_one))

    r = await comp.build_runnable(model_factory).ainvoke(StringPromptValue(text=random_string))
    assert r == random_int + 1


async def test_map(random_string: str, random_int: int) -> None:
    model_mock = MockModelRunnable()
    model_mock.return_value = str(random_int)
    model_factory: ModelRunnableFactory = lambda: model_mock

    comp = define_model().map(_to_int).map(_plus_one)

    r = await comp.build_runnable(model_factory).ainvoke(StringPromptValue(text=random_string))
    assert r == random_int + 1


async def test_contra_map(random_string: str, random_int: int) -> None:
    model_mock = MockModelRunnable()
    model_mock.return_value = str(random_int)
    model_factory: ModelRunnableFactory = lambda: model_mock

    comp = define_model().contra_map(_to_prompt_value)

    await comp.build_runnable(model_factory).ainvoke(random_string)
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, StringPromptValue)
    assert model_mock.last_input.text == random_string


async def test_dimap(random_string: str, random_int: int) -> None:
    model_mock = MockModelRunnable()
    model_mock.return_value = str(random_int)
    model_factory: ModelRunnableFactory = lambda: model_mock

    comp = define_model().dimap(_to_prompt_value, _to_int)

    r = await comp.build_runnable(model_factory).ainvoke(random_string)
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, StringPromptValue)
    assert model_mock.last_input.text == random_string
    assert r == random_int


async def test_with_trace_name(random_string: str, random_int: int) -> None:
    model_mock = MockModelRunnable()
    model_mock.return_value = str(random_int)
    model_factory: ModelRunnableFactory = lambda: model_mock

    comp = define_model().with_trace_name(random_string)

    comp_config = cast(RunnableBinding, comp.build_runnable(model_factory)).config
    assert comp_config.get("run_name") == random_string


async def test_passthrough_input(random_string: str, random_int: int) -> None:
    model_mock = MockModelRunnable()
    model_mock.return_value = str(random_int)
    model_factory: ModelRunnableFactory = lambda: model_mock

    comp = define_model().dimap(_to_prompt_value, _to_int).passthrough_input()
    r = await comp.build_runnable(model_factory).ainvoke(random_string)
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, StringPromptValue)
    assert model_mock.last_input.text == random_string
    assert r == (random_string, random_int)


async def test_passthrough_as_first(random_string: str, random_int: int, random_float: float) -> None:
    model_mock = MockModelRunnable()
    model_mock.return_value = str(random_int)
    model_factory: ModelRunnableFactory = lambda: model_mock

    comp = define_model().dimap(_to_prompt_value, _to_int).passthrough_as_first(float)
    r = await comp.build_runnable(model_factory).ainvoke((random_float, random_string))
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, StringPromptValue)
    assert model_mock.last_input.text == random_string
    assert r == (random_float, random_int)


async def test_passthrough_as_second(random_string: str, random_int: int, random_float: float) -> None:
    model_mock = MockModelRunnable()
    model_mock.return_value = str(random_int)
    model_factory: ModelRunnableFactory = lambda: model_mock

    comp = define_model().dimap(_to_prompt_value, _to_int).passthrough_as_second(float)
    r = await comp.build_runnable(model_factory).ainvoke((random_string, random_float))
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, StringPromptValue)
    assert model_mock.last_input.text == random_string
    assert r == (random_int, random_float)


async def test_pipe_to(random_string: str, random_int: int) -> None:
    model_mock = MockModelRunnable()
    model_mock.return_value = str(random_int)
    model_factory: ModelRunnableFactory = lambda: model_mock

    comp = define_model().contra_map(_to_prompt_value).pipe_to(ChainComputation.lift(_to_int)) | ChainComputation.lift(
        _plus_two
    )
    r = await comp.build_runnable(model_factory).ainvoke(random_string)
    assert model_mock.last_input is not None
    assert isinstance(model_mock.last_input, StringPromptValue)
    assert model_mock.last_input.text == random_string
    assert r == random_int + 2


async def test_in_parallel_with(random_string: str, random_int: int) -> None:
    def model_factory() -> MockModelRunnable:
        model_mock = MockModelRunnable()
        model_mock.return_value = str(random_int)
        return model_mock

    comp1 = define_model().contra_map(_to_prompt_value).pipe_to(ChainComputation.lift(_to_int)) | ChainComputation.lift(
        _plus_one
    )

    comp2 = define_model().contra_map(_to_prompt_value).pipe_to(ChainComputation.lift(_to_int)) | ChainComputation.lift(
        _plus_two
    )

    r = await comp1.in_parllel_with(comp2).build_runnable(model_factory).ainvoke((random_string, random_string))
    assert r == (random_int + 1, random_int + 2)


async def test_product(random_string: str, random_int: int) -> None:
    def model_factory() -> MockModelRunnable:
        model_mock = MockModelRunnable()
        model_mock.return_value = str(random_int)
        return model_mock

    comp1 = define_model().contra_map(_to_prompt_value).pipe_to(ChainComputation.lift(_to_int)) | ChainComputation.lift(
        _plus_one
    )

    comp2 = define_model().contra_map(_to_prompt_value).pipe_to(ChainComputation.lift(_to_int)) | ChainComputation.lift(
        _plus_two
    )

    r = await comp1.product(comp2).build_runnable(model_factory).ainvoke(random_string)
    assert r == (random_int + 1, random_int + 2)

    r = await (comp1**comp2).build_runnable(model_factory).ainvoke(random_string)
    assert r == (random_int + 1, random_int + 2)


async def test_map_n(random_string: str, random_int: int) -> None:
    def model_factory() -> MockModelRunnable:
        model_mock = MockModelRunnable()
        model_mock.return_value = str(random_int)
        return model_mock

    comp1 = define_model().dimap(_to_prompt_value, _to_int) | ChainComputation.lift(_plus_one)
    comp2 = define_model().dimap(_to_prompt_value, _to_int) | ChainComputation.lift(_plus_two)
    comp3 = define_model().dimap(_to_prompt_value, _to_float)

    comp = ChainComputation.map_n(lambda x, y, z: (x, y, z), comp1, comp2, comp3)

    r = await comp.build_runnable(model_factory).ainvoke(random_string)
    assert r == (random_int + 1, random_int + 2, float(random_int))
