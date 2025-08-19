import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from mabool.infra.operatives import UnexpectedSessionStateError
from mabool.infra.operatives.tests.mock_app import create_mock_app


@pytest.fixture
async def async_client() -> AsyncIterator[AsyncClient]:
    # without a lifespan manager, dependency injection does not work
    async with custom_async_client() as ac:
        yield ac


@asynccontextmanager
async def custom_async_client(**config_overrides: Any) -> AsyncIterator[AsyncClient]:
    # without a lifespan manager, dependency injection does not work
    mock_app = create_mock_app(**config_overrides)
    async with LifespanManager(mock_app):
        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_interactions(async_client: AsyncClient) -> None:
    url = "interactive_animals"
    dark_query = "I'm curious about how animals navigate in the dark."
    response = await async_client.post(
        url,
        headers={"Content-Type": "application/json", "Conversation-Thread-Id": "123"},
        json={"query": dark_query},
    )
    assert response.json() == {"question": "Who are you?", "options": ["fox", "bat", "box"]}

    response = await async_client.post(url, headers={"Conversation-Thread-Id": "123"}, json={"answer": "box"})
    assert response.json() == {"question": "Who are you?", "options": ["fox", "bat", "box"]}

    response = await async_client.post(url, headers={"Conversation-Thread-Id": "123"}, json={"answer": "box"})
    assert response.json() == {"question": "Who are you?", "options": ["fox", "bat", "box"]}

    response = await async_client.post(url, headers={"Conversation-Thread-Id": "123"}, json={"answer": "fox"})
    assert response.json() == {"question": "Do you have a tail?", "options": [True, False]}

    response = await async_client.post(url, headers={"Conversation-Thread-Id": "123"}, json={"answer": "false"})
    assert response.json() == {"data": ["sight", "smell", "sound"]}


@pytest.mark.asyncio
async def test_interaction_operative_error(async_client: AsyncClient) -> None:
    url = "interactive_animals"
    response = await async_client.post(
        url,
        headers={"Content-Type": "application/json", "Conversation-Thread-Id": "123"},
        json={"query": "blah"},
    )
    assert response.json() == {
        "error": {"type": "other", "message": "You have to ask about animals."},
        "analyzed_query": None,
    }


@pytest.mark.asyncio
async def test_interaction_operative_timeout() -> None:
    async with custom_async_client(operative_timeout=1) as async_client:
        url = "interactive_animals"
        with pytest.raises(asyncio.TimeoutError) as e:
            await async_client.post(
                url,
                headers={"Content-Type": "application/json", "Conversation-Thread-Id": "123"},
                json={"query": "wait"},
            )
        assert str(e.value) == "Operative/Inquire tasks of session 123 have timed out after 1 sec."


@pytest.mark.asyncio
async def test_unexpected_answer(async_client: AsyncClient) -> None:
    url = "interactive_animals"
    with pytest.raises(UnexpectedSessionStateError):
        await async_client.post(url, headers={"Conversation-Thread-Id": "123"}, json={"answer": "box"})


@pytest.mark.asyncio
async def test_no_interactions(async_client: AsyncClient) -> None:
    url = "non_interactive_animals"
    dark_query = "I'm curious about how animals navigate in the dark."
    response = await async_client.post(
        url,
        headers={"Content-Type": "application/json", "Conversation-Thread-Id": "456"},
        json={"query": dark_query},
    )
    assert response.json() == {
        "error": {"type": "other", "message": "No interactions available."},
        "analyzed_query": None,
    }
