from __future__ import annotations

from typing import Any, Sequence

from google import genai
from google.genai import types
from google.genai.types import (
    ContentListUnionDict,
    GenerateContentConfig,
    GenerateContentResponse,
    PartDict,
)
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult


class AsyncChatGoogleGenAI(BaseChatModel):
    client: genai.Client
    model_name: str
    model_kwargs: dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        return "chat-google-generative-ai"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "_type": self._llm_type,
            "model": "models/gemini-2.0-flash",
            "temperature": self.model_kwargs.get("temperature", 0),
            "top_k": self.model_kwargs.get("top_k"),
            "n": self.model_kwargs.get("candidate_count", 1),
            "safety_settings": self.model_kwargs.get("safety_settings"),
        }

    async def _agenerate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message, llm_output = await self._agenerate_message(messages, stop=stop, **kwargs)
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output=llm_output,
        )

    async def _agenerate_message(
        self, messages: Sequence[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> tuple[AIMessage, dict[str, Any]]:
        contents, system_instruction = self._format_messages(messages)
        client_config: GenerateContentConfig = self._process_model_kwargs(kwargs, stop)
        client_config.system_instruction = system_instruction
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=client_config,
        )
        response_text, usage_metadata, llm_output = self._extract_response_data(response)

        ai_message = AIMessage(content=response_text)

        if usage_metadata:
            ai_message.usage_metadata = UsageMetadata(**usage_metadata)

        return ai_message, llm_output

    def _process_model_kwargs(self, kwargs: dict[str, Any], stop: list[str] | None = None) -> GenerateContentConfig:
        merged_config = {**self.model_kwargs, **kwargs, **{"stop_sequences": stop}}
        generation_config_args = merged_config.pop("generation_config", {})
        return GenerateContentConfig.model_validate(
            {**merged_config, **generation_config_args, "automatic_function_calling": {"disable": True}},
        )

    def _format_messages(
        self, messages: Sequence[BaseMessage]
    ) -> tuple[types.ContentListUnion | types.ContentListUnionDict, str | None]:
        system_messages = []
        chat_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                system_messages.append(message)
            else:
                chat_messages.append(message)

        system_instruction = None
        if system_messages:
            system_instruction = "\n".join([msg.content for msg in system_messages])

        contents: ContentListUnionDict
        if not chat_messages:
            contents = {"role": "user", "parts": [PartDict(text="")]}
        elif len(chat_messages) == 1:
            message = chat_messages[0]
            role = "user" if isinstance(message, HumanMessage) else "model"
            contents = {"role": role, "parts": [PartDict(text=message.text())]}
        else:
            contents = []
            for message in chat_messages:
                role = self._determine_role(message)
                contents.append({"role": role, "parts": [PartDict(text=message.text())]})
        return contents, system_instruction

    def _determine_role(self, message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "model"
        else:
            return "user"

    def _extract_response_data(self, response: Any) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
        if not isinstance(response, GenerateContentResponse):
            raise ValueError("Unexpected response format, streaming currently not supported")

        llm_output = self._extract_llm_metadata(response)
        usage_metadata = self._extract_usage_metadata(response)
        text = self._extract_text_from_response(response, llm_output)

        return text, usage_metadata, llm_output

    def _extract_llm_metadata(self, response: GenerateContentResponse) -> dict[str, Any]:
        llm_output = {}

        llm_output["model_version"] = response.model_version
        llm_output["prompt_feedback"] = response.prompt_feedback

        candidates = response.candidates
        if candidates:
            candidate = candidates[0]
            llm_output["finish_reason"] = candidate.finish_reason
            llm_output["safety_ratings"] = candidate.safety_ratings
            llm_output["avg_logprobs"] = candidate.avg_logprobs

        return llm_output

    def _extract_usage_metadata(self, response: GenerateContentResponse) -> dict[str, Any] | None:
        if not response.usage_metadata:
            return None

        usage_metadata_obj = response.usage_metadata

        input_tokens = usage_metadata_obj.prompt_token_count or 0
        output_tokens = usage_metadata_obj.candidates_token_count or 0
        total_tokens = usage_metadata_obj.total_token_count or 0

        if input_tokens + output_tokens + total_tokens == 0:
            return None

        usage_metadata: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

        cached_token_count = usage_metadata_obj.cached_content_token_count
        if cached_token_count:
            usage_metadata["input_token_details"] = {"cache_read": cached_token_count}

        return usage_metadata

    def _extract_text_from_response(self, response: GenerateContentResponse, llm_output: dict) -> str:
        candidates = response.candidates or []
        if not candidates:
            llm_output["error"] = "No candidates returned"
            return ""

        try:
            candidate = candidates[0]
            content = candidate.content
            if content and content.parts:
                return content.parts[0].text or ""
            return ""
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to extract text from response: {e}")

    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError("_acall is not implemented for this model.")

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError("Synchronous calls are not supported. Use _acall instead.")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("Synchronous generation is not supported. Use _agenerate instead.")
