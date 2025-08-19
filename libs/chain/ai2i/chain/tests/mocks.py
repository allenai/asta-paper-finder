from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig


@dataclass
class MockModelRunnable(Runnable[PromptValue, BaseMessage]):
    return_value: str | None = field(default=None, init=False)
    return_metadata: dict[str, Any] = field(default_factory=dict, init=False)
    last_input: PromptValue | None = field(default=None, init=False)
    throw_error_n_times: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        super().__init__()

    async def ainvoke(self, input: PromptValue, config: RunnableConfig | None = None, **kwargs: Any) -> BaseMessage:
        self.last_input = input

        if self.throw_error_n_times > 0:
            self.throw_error_n_times -= 1
            raise ValueError(f"Mock Error, {self.throw_error_n_times} throws left")

        if self.return_value is not None:
            return BaseMessage(content=self.return_value, type="ChatGeneration", response_metadata=self.return_metadata)
        else:
            raise ValueError("No return_value defined for this mock")

    def invoke(self, input: PromptValue, config: RunnableConfig | None = None, **kwargs: Any) -> BaseMessage:
        raise NotImplementedError(f"No support for blocking calls in {self.__class__.__name__}")
