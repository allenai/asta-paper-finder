from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    Sequence,
    Type,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
    overload,
)

from langchain.output_parsers.fix import OutputFixingParser
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables.base import Runnable, RunnableLambda
from pydantic import BaseModel

from ai2i.chain.computation import ChainComputation, ModelRunnableFactory

logger = logging.getLogger(__name__)


# because we can't define a typevar that is bound directly by TypedDict, this is the workaround
# we define an "empty" typeddict, and because the typeddict is checked by implicit inheritance (like in typescript)
# all typedict that don't even explicitly inherit from this type will be considered correct type assignments
class AnyTypedDict(TypedDict):
    pass


PROMPT_PARAMS = TypeVar("PROMPT_PARAMS", bound=AnyTypedDict)
LLM_STRUCTURED_OUTPUT = TypeVar("LLM_STRUCTURED_OUTPUT", bound=BaseModel)
LLM_OUTPUT = TypeVar("LLM_OUTPUT")


# these are the types that are currently supported by langchain
# NOTE: Not including jinja intentionally, we don't use it currently and eventually
#       we will use a single templeting engine
TemplateFormat: TypeAlias = Literal["f-string", "mustache"]

TrueLiteral = Literal[True]
FalseLiteral = Literal[False]

ResponseMetadata = dict[str, Any]


# The following overload changes the return type based on the 'include_response_metadata' parameter.
# If it's set to 'True', the result will contain the output and a ResponseMetadata dict, if it's set to
# 'False', it will only include the output
@overload
def define_prompt_llm_call(
    template: str,
    /,
    input_type: Type[PROMPT_PARAMS],
    output_type: Type[LLM_STRUCTURED_OUTPUT],
    include_response_metadata: TrueLiteral,
    format: TemplateFormat = "f-string",
    get_extra_params: Callable[[], Mapping[str, str]] = dict,
    custom_format_instructions: str | None = None,
) -> ChainComputation[PROMPT_PARAMS, tuple[LLM_STRUCTURED_OUTPUT, ResponseMetadata]]:
    pass


@overload
def define_prompt_llm_call(
    template: str,
    /,
    input_type: Type[PROMPT_PARAMS],
    output_type: Type[LLM_STRUCTURED_OUTPUT],
    include_response_metadata: FalseLiteral = False,
    format: TemplateFormat = "f-string",
    get_extra_params: Callable[[], Mapping[str, str]] = dict,
    custom_format_instructions: str | None = None,
) -> ChainComputation[PROMPT_PARAMS, LLM_STRUCTURED_OUTPUT]:
    pass


def define_prompt_llm_call(
    template: str,
    /,
    input_type: Type[PROMPT_PARAMS],
    output_type: Type[LLM_STRUCTURED_OUTPUT],
    include_response_metadata: bool = False,
    format: TemplateFormat = "f-string",
    get_extra_params: Callable[[], Mapping[str, str]] = dict,
    custom_format_instructions: str | None = None,
) -> ChainComputation[PROMPT_PARAMS, LLM_STRUCTURED_OUTPUT | tuple[LLM_STRUCTURED_OUTPUT, ResponseMetadata]]:
    output_parser = PydanticOutputParser(pydantic_object=output_type)

    format_instructions = (
        custom_format_instructions
        if custom_format_instructions is not None
        else output_parser.get_format_instructions()
    )

    # NOTE: we lstrip() the template to make it align better in code (can start from a new line)
    template_with_format: str
    if format == "f-string":
        template_with_format = template.lstrip() + "\n{format_instructions}"
    else:  # format == mustache
        template_with_format = template.lstrip() + "\n{{format_instructions}}"

    # suspend the creation of the prompt, because we want the 'get_extra_params' to be called when application
    # is running and not during the define (otherwise there is no config in context fo the get_extra_params to read it)
    create_template: ChainComputation[PROMPT_PARAMS, PromptValue] = ChainComputation.suspend_runnable(
        lambda: cast(
            Runnable[PROMPT_PARAMS, PromptValue],
            PromptTemplate(
                template=template_with_format,
                template_format=format,
                input_variables=list(input_type.__annotations__),
                input_types=input_type.__annotations__,
                partial_variables={"format_instructions": format_instructions, **get_extra_params()},
            ),
        )
    )

    create_output: ChainComputation[BaseMessage, LLM_STRUCTURED_OUTPUT | tuple[LLM_STRUCTURED_OUTPUT, ResponseMetadata]]
    if include_response_metadata:
        create_output = (
            define_parser_with_llm(parser=output_parser)
            .passthrough_input()
            .map(lambda t: (t[1], cast(dict[str, Any], t[0].response_metadata)))
        )
    else:
        create_output = define_parser_with_llm(parser=output_parser)

    return create_template | define_model() | create_output


ChatUser = Literal["system", "user", "ai"]
ChatMessage = tuple[ChatUser, str]


def system_message(text: str) -> ChatMessage:
    return ("system", text)


def user_message(text: str) -> ChatMessage:
    return ("user", text)


def assistant_message(text: str) -> ChatMessage:
    return ("ai", text)


# The following overload changes the return type based on the 'include_response_metadata' parameter.
# If it's set to 'True', the result will contain the output and a ResponseMetadata dict, if it's set to
# 'False', it will only include the output
@overload
def define_chat_llm_call(
    messages: Sequence[ChatMessage],
    /,
    input_type: Type[PROMPT_PARAMS],
    output_type: Type[LLM_STRUCTURED_OUTPUT],
    include_response_metadata: TrueLiteral,
    format: TemplateFormat = "f-string",
    get_extra_params: Callable[[], Mapping[str, str]] = dict,
) -> ChainComputation[PROMPT_PARAMS, tuple[LLM_STRUCTURED_OUTPUT, ResponseMetadata]]:
    pass


@overload
def define_chat_llm_call(
    messages: Sequence[ChatMessage],
    /,
    input_type: Type[PROMPT_PARAMS],
    output_type: Type[LLM_STRUCTURED_OUTPUT],
    include_response_metadata: FalseLiteral = False,
    format: TemplateFormat = "f-string",
    get_extra_params: Callable[[], Mapping[str, str]] = dict,
) -> ChainComputation[PROMPT_PARAMS, LLM_STRUCTURED_OUTPUT]:
    pass


def define_chat_llm_call(
    messages: Sequence[ChatMessage],
    /,
    input_type: Type[PROMPT_PARAMS],
    output_type: Type[LLM_STRUCTURED_OUTPUT],
    include_response_metadata: bool = False,
    format: TemplateFormat = "f-string",
    get_extra_params: Callable[[], Mapping[str, str]] = dict,
) -> ChainComputation[PROMPT_PARAMS, LLM_STRUCTURED_OUTPUT | tuple[LLM_STRUCTURED_OUTPUT, ResponseMetadata]]:
    output_parser = PydanticOutputParser(pydantic_object=output_type)
    # we lstrip all messages, for better alignment in multiline strings
    lstripped_messages = [(t, m.lstrip()) for t, m in messages]
    chat_template = ChatPromptTemplate.from_messages(lstripped_messages, template_format=format)

    create_template: ChainComputation[PROMPT_PARAMS, PromptValue] = ChainComputation.lift(
        cast(Runnable[PROMPT_PARAMS, PromptValue], chat_template)
    )

    create_output: ChainComputation[BaseMessage, LLM_STRUCTURED_OUTPUT | tuple[LLM_STRUCTURED_OUTPUT, ResponseMetadata]]
    if include_response_metadata:
        create_output = (
            define_parser_with_llm(parser=output_parser)
            .passthrough_input()
            .map(lambda t: (t[1], cast(dict[str, Any], t[0].response_metadata)))
        )
    else:
        create_output = define_parser_with_llm(parser=output_parser)

    return (
        _enrich_input_with(input_type, get_extra_params).with_trace_name("add extra_params to prompt")
        | create_template
        | define_model()
        | create_output
    )


def _enrich_input_with(
    input_type: Type[PROMPT_PARAMS], f: Callable[[], Mapping[str, Any]]
) -> ChainComputation[PROMPT_PARAMS, PROMPT_PARAMS]:
    def _internal_enrich_input_with(d: PROMPT_PARAMS) -> PROMPT_PARAMS:
        return cast(PROMPT_PARAMS, {**d, **f()})

    return ChainComputation.lift(_internal_enrich_input_with)


def define_model() -> ChainComputation[PromptValue, BaseMessage]:
    def _internal_define_model(mf: ModelRunnableFactory) -> Runnable[PromptValue, BaseMessage]:
        return mf()

    return ChainComputation(_internal_define_model)


def define_parser_with_llm(
    parser: PydanticOutputParser[LLM_STRUCTURED_OUTPUT],
) -> ChainComputation[BaseMessage, LLM_STRUCTURED_OUTPUT]:
    def _create_retry_logger() -> ChainComputation[LLM_OUTPUT, LLM_OUTPUT]:
        def log_retry(inputs: LLM_OUTPUT) -> LLM_OUTPUT:
            logger.warning(f"Parser retry attempt: {str(inputs)}")
            return inputs

        return ChainComputation.lift(log_retry)

    def _internal_define_parser_with_llm(mf: ModelRunnableFactory) -> Runnable[BaseMessage, LLM_STRUCTURED_OUTPUT]:
        output_fixing_parser = OutputFixingParser.from_llm(
            llm=_create_retry_logger().build_runnable(mf) | mf(), parser=parser
        )

        # NOTE: This code below is required because the OutputFixingParser (defined above)
        # will call the LLM's abstraction `invoke` method instead of `ainvoke`,
        # even if the abstraction was initially called with `ainvoke`.
        async def parse(input_message: BaseMessage) -> LLM_STRUCTURED_OUTPUT:
            if isinstance(input_message.content, str):
                return cast(LLM_STRUCTURED_OUTPUT, await output_fixing_parser.aparse(input_message.content))
            raise ValueError(f"BaseMessage's content is of type {type(input_message.content)}, which is unsupported.")

        return RunnableLambda(parse)

    return ChainComputation(_internal_define_parser_with_llm)
