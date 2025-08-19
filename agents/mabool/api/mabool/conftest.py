import asyncio
from pathlib import Path
from typing import Iterator

import pytest
from ai2i.config import application_config_ctx, load_conf


@pytest.fixture(scope="function", autouse=True)
def global_config() -> Iterator[None]:
    with application_config_ctx(load_conf(Path(__file__).parent.parent / "conf")):
        yield None


# Designed to take care of event loop closed errors.
@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    loop.slow_callback_duration = 1.0
    yield loop
    loop.close()
