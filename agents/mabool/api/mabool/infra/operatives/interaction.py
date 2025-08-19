from __future__ import annotations

import asyncio
import uuid
from asyncio import Task
from functools import wraps
from typing import Any, Callable, Protocol

from ai2i.config import ConfigValue, configurable
from ai2i.di import DI, RoundId, builtin_deps
from anyio import create_memory_object_stream

from mabool.data_model.config import cfg_schema
from mabool.data_model.ids import ConversationThreadId
from mabool.infra.operatives.operatives import (
    INPUT,
    OUTPUT,
    STATE,
    InquiryQuestion,
    InquiryReply,
    Interactions,
    Operative,
    OperativeResponse,
    ReplyAndScopesContext,
    SessionData,
    StateManager,
    interaction_manager,
    session_context,
    session_context_middleware,
)


class DecoratorFuncType[INPUT, OUTPUT](Protocol):
    async def __call__(
        self, query: INPUT | InquiryReply, conversation_thread_id: ConversationThreadId
    ) -> OperativeResponse[OUTPUT] | InquiryQuestion | InquiryReply: ...


class InteractionTimeoutError(asyncio.TimeoutError):
    def __init__(self, session_id: str, timeout: float) -> None:
        super().__init__(f"Operative/Inquire tasks of session {session_id} have timed out after {timeout} sec.")
        self.session_id = session_id
        self.timeout = timeout


def operative_session(
    operative_factory: Callable[[StateManager[STATE], Interactions | None], Operative[INPUT, OUTPUT, STATE]],
    operative_id: str,
    allow_interactions: bool = True,
) -> Callable[[DecoratorFuncType[INPUT, OUTPUT]], DecoratorFuncType[INPUT, OUTPUT]]:
    def decorator(fn: DecoratorFuncType[INPUT, OUTPUT]) -> DecoratorFuncType[INPUT, OUTPUT]:
        operative_instance: Operative[INPUT, OUTPUT, STATE] = operative_factory(
            StateManager.init_root_manager(operative_id), None
        )

        @wraps(fn)
        @DI.managed
        async def wrap(
            query: INPUT | InquiryReply,
            conversation_thread_id: ConversationThreadId,
            round_id: RoundId | None = DI.requires(builtin_deps.round_id, default=None),
        ) -> OperativeResponse[OUTPUT] | InquiryQuestion | InquiryReply:
            state_id = round_id or str(uuid.uuid4())

            with session_context_middleware(conversation_thread_id, state_id):
                session_id = await _init_interactive_session(allow_interactions)

                if isinstance(query, InquiryReply):
                    return await _handle_inquiry_reply(query, session_id)

                return await _run_operative(operative_instance, session_id, query)

        return wrap

    return decorator


async def _init_interactive_session(allow_interactions: bool = True) -> str:
    session_id = session_context.get()
    if session_id not in interaction_manager:
        inquiry_send_stream, inquiry_receive_stream = create_memory_object_stream[InquiryQuestion]()
        inquiry_response_send_stream, inquiry_response_receive_stream = create_memory_object_stream[
            ReplyAndScopesContext
        ]()

        class DefaultInteractions(Interactions):
            async def ask(self, inquiry: InquiryQuestion) -> InquiryReply:
                await interaction_manager.send_inquiry(session_id, inquiry)
                interactive_reply = await interaction_manager.receive_reply(session_id)
                return InquiryReply(answer=interactive_reply.answer)

        interaction_manager[session_id] = SessionData(
            inquiry_send_channel=inquiry_send_stream,
            inquiry_receive_channel=inquiry_receive_stream,
            inquiry_reply_send_channel=inquiry_response_send_stream,
            inquiry_reply_receive_channel=inquiry_response_receive_stream,
            interactions=DefaultInteractions() if allow_interactions else None,
        )
    return session_id


async def _run_operative(
    operative: Operative,
    session_id: str,
    query: Any,
) -> OperativeResponse[OUTPUT] | InquiryQuestion:
    def clear_state(task: Task) -> None:
        operative.clear_state()
        interaction_manager[session_id].pending_tasks.discard(task)

    operative_task = asyncio.create_task(operative(query))
    operative_task.add_done_callback(clear_state)
    return await _create_and_wait_for_tasks(session_id, operative_task)


async def _handle_inquiry_reply(query: InquiryReply, session_id: str) -> OperativeResponse[OUTPUT] | InquiryQuestion:
    await interaction_manager.send_reply(session_id, query)
    operative_task = interaction_manager.pop_pending_task(session_id)
    return await _create_and_wait_for_tasks(session_id, operative_task)


@configurable
async def _create_and_wait_for_tasks(
    session_id: str,
    operative_task: asyncio.Task[OperativeResponse[OUTPUT]],
    timeout: float = ConfigValue(cfg_schema.operative_timeout),
) -> OperativeResponse[OUTPUT] | InquiryQuestion:
    current_session = interaction_manager[session_id]

    async def wait_for_inquiry() -> InquiryQuestion:
        return await interaction_manager.receive_inquiry(session_id)

    inquire_task = asyncio.create_task(wait_for_inquiry())
    current_session.pending_tasks.add(operative_task)
    done_tasks, pending_tasks = await asyncio.wait(
        [inquire_task, operative_task], return_when=asyncio.FIRST_COMPLETED, timeout=timeout
    )

    await _finalize_pending_tasks(
        pending_tasks=pending_tasks,
        session_id=session_id,
        is_timeout=not done_tasks,
        is_operative_done=operative_task in done_tasks,
        timeout=timeout,
    )

    return done_tasks.pop().result()


async def _finalize_pending_tasks(
    pending_tasks: set[asyncio.Task], session_id: str, is_timeout: bool, is_operative_done: bool, timeout: float
) -> None:
    if is_timeout:
        [pending_task.cancel() for pending_task in pending_tasks]
        await asyncio.wait(pending_tasks)
        raise InteractionTimeoutError(session_id=session_id, timeout=timeout)
    if is_operative_done:
        [pending_task.cancel() for pending_task in pending_tasks]
        await asyncio.wait(pending_tasks)
