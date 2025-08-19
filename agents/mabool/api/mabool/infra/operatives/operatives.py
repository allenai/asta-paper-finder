from __future__ import annotations

import asyncio
import logging
import uuid
from abc import abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator, Type, TypeAlias, TypeVar

from ai2i.di import DI, ApplicationContext, ApplicationScopes, builtin_deps
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from cachetools import TTLCache
from pydantic import BaseModel, Field

from mabool.data_model.agent import AgentError, AnalyzedQuery, PartiallyAnalyzedQuery

SessionId: TypeAlias = str
StateId: TypeAlias = str
session_context: ContextVar[SessionId] = ContextVar("session_context")
state_context: ContextVar[StateId] = ContextVar("state_context")

INPUT = TypeVar("INPUT", contravariant=True)
NESTED_INPUT = TypeVar("NESTED_INPUT", contravariant=True)
OUTPUT = TypeVar("OUTPUT", covariant=True)
NESTED_OUTPUT = TypeVar("NESTED_OUTPUT", covariant=True)
STATE = TypeVar("STATE")
NESTED_STATE = TypeVar("NESTED_STATE")
OPERATIVE = TypeVar("OPERATIVE", bound="Operative")

logger = logging.getLogger(__name__)


class SessionData(BaseModel):
    inquiry_send_channel: MemoryObjectSendStream[InquiryQuestion]
    inquiry_receive_channel: MemoryObjectReceiveStream[InquiryQuestion]

    inquiry_reply_send_channel: MemoryObjectSendStream[ReplyAndScopesContext]
    inquiry_reply_receive_channel: MemoryObjectReceiveStream[ReplyAndScopesContext]

    pending_tasks: set[asyncio.Task] = Field(default_factory=set)
    interactions: Interactions | None = Field(default=None)

    class Config:
        arbitrary_types_allowed = True


class InteractionManager:
    _sessions: ClassVar[TTLCache[SessionId, SessionData]] = TTLCache[SessionId, SessionData](maxsize=10000, ttl=86400)

    def __getitem__(self, session_id: SessionId) -> SessionData:
        return self._sessions[session_id]

    def __setitem__(self, session_id: SessionId, session_data: SessionData) -> None:
        self._sessions[session_id] = session_data

    def __contains__(self, session_id: SessionId) -> bool:
        return session_id in self._sessions

    @DI.managed
    async def send_reply(
        self,
        session_id: SessionId,
        response: InquiryReply,
        app_ctx: ApplicationContext = DI.requires(builtin_deps.application_context),
    ) -> None:
        logger.debug("Session ID: " + session_id + " Sending InteractiveReply: " + response.answer)
        try:
            async with asyncio.timeout(1):
                # NOTE: we send this "logical thread" DI scopes context, so that the background operative
                #       thread can load the correct DI scope, for it to be available in the operatives code
                #       without this, the background thread will still hold the context of the first request
                #       that has expired already
                await self._sessions[session_id].inquiry_reply_send_channel.send(
                    ReplyAndScopesContext(response, app_ctx.scopes)
                )
        except asyncio.TimeoutError:
            error_msg = (
                f"Session ID: {session_id} Timeout while sending InteractiveReply. "
                f"Most likely this is due to an answer being received when unexpected."
            )
            logger.exception(error_msg)
            raise UnexpectedSessionStateError(error_msg)

    @DI.managed
    async def receive_reply(
        self,
        session_id: SessionId,
        app_ctx: ApplicationContext = DI.requires(builtin_deps.application_context),
    ) -> InquiryReply:
        logger.debug("Session ID: " + session_id + " Receiving InteractiveReply")
        inquiry_and_context = await self._sessions[session_id].inquiry_reply_receive_channel.receive()
        inquiry = inquiry_and_context.reply
        app_ctx.scopes = inquiry_and_context.scopes
        logger.debug("Session ID: " + session_id + " Received InteractiveReply: " + inquiry.answer)
        return inquiry

    async def send_inquiry(self, session_id: SessionId, inquiry: InquiryQuestion) -> None:
        logger.debug("Session ID: " + session_id + " Sending Inquiry: " + inquiry.question)
        await self._sessions[session_id].inquiry_send_channel.send(inquiry)

    async def receive_inquiry(self, session_id: SessionId) -> InquiryQuestion:
        logger.debug("Session ID: " + session_id + " Receiving Inquiry")
        inquiry = await self._sessions[session_id].inquiry_receive_channel.receive()
        logger.debug("Session ID: " + session_id + " Received Inquiry: " + inquiry.question)
        return inquiry

    def pop_pending_task(self, session_id: SessionId) -> asyncio.Task:
        return self._sessions[session_id].pending_tasks.pop()

    def interactions(self) -> Interactions | None:
        return self._sessions[session_context.get()].interactions


interaction_manager = InteractionManager()


class VoidResponse(BaseModel):
    error: AgentError
    analyzed_query: AnalyzedQuery | PartiallyAnalyzedQuery | None = None


class PartialResponse[OUTPUT](BaseModel):
    data: OUTPUT
    error: AgentError | None


class CompleteResponse[OUTPUT](BaseModel):
    data: OUTPUT


class UnexpectedSessionStateError(Exception):
    pass


type OperativeResponse[T] = VoidResponse | PartialResponse[T] | CompleteResponse[T]

type StateManagerId = str


# State.
@dataclass
class StateManager[STATE]:
    # This is a simple in-memory state manager, can later be replaced with a persistent storage.
    _state_dict: ClassVar[TTLCache[tuple[StateManagerId, SessionId], Any]] = TTLCache[
        tuple[StateManagerId, SessionId], STATE
    ](maxsize=10000, ttl=86400)

    id: StateManagerId = field()
    _child_state_managers: dict[StateManagerId, StateManager[STATE]] = field(default_factory=dict)

    def get_state(self) -> STATE | None:
        return self._state_dict.get((self.id, state_context.get()))

    def set_state(self, state: STATE | None) -> None:
        if state:
            self._state_dict[(self.id, state_context.get())] = state

    def clear_state(self) -> None:
        for child_state_manager in self._child_state_managers.values():
            child_state_manager.clear_state()
        self._state_dict.pop((self.id, state_context.get()), None)

    def init_manager(self, id: StateManagerId) -> StateManager[Any]:
        child_state_manager = StateManager(f"{self.id}:{id}")
        self._child_state_managers[id] = child_state_manager
        return child_state_manager

    @staticmethod
    def init_root_manager(id: StateManagerId) -> StateManager[STATE]:
        return StateManager[STATE](id=id)


# Interactions.
InteractiveInquiryOption = list[str] | list[int] | list[bool]


class InquiryQuestion(BaseModel):
    question: str
    options: InteractiveInquiryOption = Field(default_factory=list)


class InquiryReply(BaseModel):
    answer: str


# NOTE: this will never go over the wire, so it doesn't need to be a BaseModel
@dataclass(frozen=True)
class ReplyAndScopesContext:
    reply: InquiryReply
    scopes: ApplicationScopes


class Interactions(BaseModel):
    @abstractmethod
    async def ask(self, inquiry: InquiryQuestion) -> InquiryReply:
        # abstract.
        ...


class Inquiry(BaseModel):
    interactions: Interactions

    async def ask(self, question: InquiryQuestion) -> InquiryReply:
        return await self.interactions.ask(question)


# Operative.
class Operative[INPUT, OUTPUT, STATE]:
    def __init__(self, state_manager: StateManager[STATE] | None = None, interactions: Interactions | None = None):
        self._interactions = interactions
        self._state_manager = state_manager or StateManager[STATE].init_root_manager(
            f"{self.__class__.__name__}_{uuid.uuid4().hex}"
        )
        self.register()

    async def __call__(self, inputs: INPUT) -> OperativeResponse[OUTPUT]:
        original_state = self._state_manager.get_state()
        try:
            state, response = await self.handle_operation(original_state, inputs)
            self._state_manager.set_state(state)
            return response
        except Exception as e:
            logger.exception(f"An error occurred while running {self.__class__.__name__}: {e}")
            return VoidResponse(error=AgentError(type="other", message=str(e)))

    def inquiry(self) -> Inquiry | None:
        interactions = interaction_manager.interactions() if not self._interactions else self._interactions
        if not interactions:
            return None
        return Inquiry(interactions=interactions)

    def init_operative(
        self,
        id: str,
        operative_type: Type[OPERATIVE],
        interactions: Interactions | None = None,
    ) -> OPERATIVE:
        return operative_type(self._state_manager.init_manager(id), interactions)

    def register(self) -> None:
        # implement to register sub-operatives within the lifecycle of the parent operative.
        ...

    def clear_state(self) -> None:
        self._state_manager.clear_state()

    @abstractmethod
    async def handle_operation(
        self, state: STATE | None, inputs: INPUT
    ) -> tuple[STATE | None, OperativeResponse[OUTPUT]]:
        # abstract.
        ...


@contextmanager
def session_context_middleware(conversation_thread_id: str, state_id: StateId) -> Iterator[None]:
    session_context.set(conversation_thread_id)
    state_context.set(state_id)
    yield
