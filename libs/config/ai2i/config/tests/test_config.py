from ai2i.config import (
    ConfigValue,
    config_value,
    configurable,
    with_config_overrides,
)
from ai2i.config.tests.config import cfg_schema


@with_config_overrides(
    run_snowball_for_central=True,
    dense_agent={"initial_top_k_per_query": 50, "reformulate_prompt_example_docs": 20},
)
def test_overrides_dict() -> None:
    assert config_value(cfg_schema.dense_agent.initial_top_k_per_query) == 50
    assert config_value(cfg_schema.dense_agent.reformulate_prompt_example_docs) == 20
    assert config_value(cfg_schema.run_snowball_for_central)


@with_config_overrides(
    run_snowball_for_central=False,
    dense_agent={"initial_top_k_per_query": 50, "reformulate_prompt_example_docs": 20},
)
async def test_overrides_dict_async() -> None:
    assert config_value(cfg_schema.dense_agent.initial_top_k_per_query) == 50
    assert config_value(cfg_schema.dense_agent.reformulate_prompt_example_docs) == 20
    assert not config_value(cfg_schema.run_snowball_for_central)


def test_defaults() -> None:
    assert config_value(cfg_schema.dense_agent.initial_top_k_per_query) == 2
    assert config_value(cfg_schema.dense_agent.reformulate_prompt_example_docs) == 3


def test_config_decorator_simple_func() -> None:
    @configurable
    def get_value(k: int = ConfigValue(cfg_schema.dense_agent.initial_top_k_per_query)) -> int:
        return k

    assert get_value() == 2
    assert get_value(19) == 19


def test_config_decorator_simple_kwarg_only_func() -> None:
    @configurable
    def get_value(*, k: int = ConfigValue(cfg_schema.dense_agent.initial_top_k_per_query)) -> int:
        return k

    assert get_value() == 2
    assert get_value(k=19) == 19


def test_config_decorator_many_args() -> None:
    @configurable
    def get_value(n: str, k: int = ConfigValue(cfg_schema.dense_agent.initial_top_k_per_query)) -> tuple[str, int]:
        return n, k

    assert get_value("AB") == ("AB", 2)
    assert get_value("AB", 19) == ("AB", 19)


def test_config_decorator_many_args_other_defaults() -> None:
    @configurable
    def get_value(
        n: str = "GOOD", k: int = ConfigValue(cfg_schema.dense_agent.initial_top_k_per_query)
    ) -> tuple[str, int]:
        return n, k

    assert get_value() == ("GOOD", 2)
    assert get_value("OTHER") == ("OTHER", 2)
    assert get_value("OTHER", 19) == ("OTHER", 19)


def test_config_decorator_many_args_complex_defaults() -> None:
    @configurable
    def get_value(
        s: str, n: str = "GOOD", /, k: int = ConfigValue(cfg_schema.dense_agent.initial_top_k_per_query)
    ) -> tuple[str, str, int]:
        return s, n, k

    assert get_value("FIRST") == ("FIRST", "GOOD", 2)
    assert get_value("FIRST", "OTHER") == ("FIRST", "OTHER", 2)
    assert get_value("FIRST", "OTHER", k=19) == ("FIRST", "OTHER", 19)


def test_config_decorator_many_args_complex_defaults2() -> None:
    @configurable
    def get_value(
        s: str,
        k: int = ConfigValue(cfg_schema.dense_agent.initial_top_k_per_query),
        /,
        n: str = "GOOD",
    ) -> tuple[str, str, int]:
        return s, n, k

    assert get_value("FIRST") == ("FIRST", "GOOD", 2)
    assert get_value("FIRST", 19) == ("FIRST", "GOOD", 19)
    assert get_value("FIRST", 19, n="OTHER") == ("FIRST", "OTHER", 19)
