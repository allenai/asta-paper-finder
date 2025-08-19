from typing import TypedDict

from ai2i.chain import define_prompt_llm_call
from pydantic import BaseModel

# -------------------------- #
# Disambiguate User Response #
# -------------------------- #


_disambiguate_user_response_prompt_tmpl = """
Given the following Question, to which of the given answer Options is the given Answer refers to:

Question:
```
{agents_question}
```

Options: ```{options}```

Input: ```{user_response}```
"""  # noqa: E501


class DisambiguateUserResponseInput(TypedDict):
    agents_question: str
    options: list[str]
    user_response: str


class DisambiguateUserResponseOutput(BaseModel):
    answer_index: int


disambiguate_user_response = define_prompt_llm_call(
    _disambiguate_user_response_prompt_tmpl,
    input_type=DisambiguateUserResponseInput,
    output_type=DisambiguateUserResponseOutput,
    custom_format_instructions=(
        'Return a JSON dict with the key "answer_index" and the value is a int representing the user answered index.'
    ),
).map(lambda o: o.answer_index)
