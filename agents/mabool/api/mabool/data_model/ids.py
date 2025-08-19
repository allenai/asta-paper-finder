import uuid
from typing import NewType

ConversationThreadId = NewType("ConversationThreadId", str)


def generate_conversation_thread_id() -> ConversationThreadId:
    return ConversationThreadId(f"thrd:{str(uuid.uuid4())}")
