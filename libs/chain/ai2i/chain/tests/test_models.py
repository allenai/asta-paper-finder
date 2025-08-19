import random

import pytest

from ai2i.chain.models import DEFAULT_N, LLMModel


@pytest.fixture
def random_float() -> float:
    return random.random()


def test_model_for_name() -> None:
    assert LLMModel.from_name("openai:gpt4-default") == LLMModel.gpt4()
    assert LLMModel.from_name("openai:gpt4turbo-default") == LLMModel.gpt4turbo()
    assert LLMModel.from_name("openai:gpt4o-default") == LLMModel.gpt4o()


def test_overriding_of_params(random_float: float) -> None:
    model = LLMModel.gpt4().override(temperature=random_float)

    assert model.params.get("temperature") == random_float
    assert model.params.get("n") == DEFAULT_N
    assert model.name == LLMModel.gpt4().name
