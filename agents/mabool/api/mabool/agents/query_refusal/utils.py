import logging
from typing import Literal, assert_never

from ai2i.chain import (
    ChainComputation,
    LLMModel,
    Timeouts,
    define_chat_llm_call,
    define_llm_endpoint,
    system_message,
    user_message,
)
from ai2i.config import config_value, ufv
from pydantic import BaseModel

from mabool.agents.common.common import InputQuery, as_input_query
from mabool.agents.common.explain import explain_query_analysis
from mabool.data_model.agent import AnalyzedQuery, PartiallyAnalyzedQuery
from mabool.data_model.config import cfg_schema
from mabool.data_model.ufs import uf
from mabool.infra.operatives import Inquiry, InquiryQuestion
from mabool.utils.llm_utils import get_api_key_for_model
from mabool.utils.text import HRULE, join_paragraphs

logger = logging.getLogger(__name__)


refusal_requires_clarification = {
    "similar to": False,
    "web access": False,
    "not paper finding": True,
    "affiliation": True,
    "author ID": True,
}


def refusal_responses() -> dict[str, str]:
    return {
        "similar to": ufv(uf.response_texts.refusal.similar_to),
        "web access": ufv(uf.response_texts.refusal.web_access),
        "not paper finding": ufv(uf.response_texts.refusal.not_paper_finding),
        "affiliation": ufv(uf.response_texts.refusal.affiliation),
        "author ID": ufv(uf.response_texts.refusal.author_id),
    }


class Choice(BaseModel):
    choice: bool | None


_choice_prompt = """
The user is asked to answer a yes-no question, "should the agent run paper search anyway?"

Given a query representing an answer to the above question,
return a JSON object that looks like {"choice": boolean or null}.
If the query represents an ambiguous choice that could be translated into yes or no, reply {"choice": null}.

Examples:
If the query is "yes" or "True", return {"choice": True}.
If the query is "cancel", return {"choice": False}.
If the query is "no, please try this new query...", return {"choice": null}.
"""

choice: ChainComputation[str, bool | None] = (
    define_chat_llm_call(
        [system_message(_choice_prompt), user_message("{{query}}")],
        format="mustache",
        input_type=InputQuery,
        output_type=Choice,
    )
    .contra_map(as_input_query)
    .map(lambda o: o.choice)
)


async def clarify_query_refusal(
    inquiry: Inquiry | None, analyzed_input: AnalyzedQuery | PartiallyAnalyzedQuery
) -> Literal["run_anyway", "cancel"]:
    async def ask_for_option(inquiry: Inquiry, attempts: int) -> str:
        if attempts == 0:
            # if we got here we have a refusal and a valid type
            assert analyzed_input.possible_refusal.type is not None

            reply = await inquiry.ask(
                InquiryQuestion(
                    question=join_paragraphs(
                        refusal_responses()[analyzed_input.possible_refusal.type],
                        HRULE,
                        ufv(uf.response_texts.refusal.came_up),
                        *explain_query_analysis(analyzed_input),
                        HRULE,
                        ufv(uf.response_texts.refusal.should_run),
                    ),
                )
            )
        else:
            reply = await inquiry.ask(InquiryQuestion(question=ufv(uf.response_texts.refusal.yes_or_no)))
        return reply.answer

    async def convert_choice_to_valid_number(user_choice: str) -> bool | None:
        llm_model = LLMModel.from_name(config_value(cfg_schema.query_analyzer_agent.llm_abstraction_model_name))
        return (
            await define_llm_endpoint(
                default_timeout=Timeouts.short,
                default_model=llm_model,
                logger=logger,
                api_key=get_api_key_for_model(llm_model),
            )
            .model_params(temperature=0)
            .execute(choice)
            .once(user_choice)
        )

    if inquiry is None:
        return "run_anyway"

    attempts = 0
    answer = None
    while answer is None:
        user_choice = await ask_for_option(inquiry, attempts)
        answer = await convert_choice_to_valid_number(user_choice)
        attempts += 1
    match answer:
        case True:
            return "run_anyway"
        case False:
            return "cancel"
        case never:
            raise assert_never(never)
