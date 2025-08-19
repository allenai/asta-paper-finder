from __future__ import annotations

from ai2i.di import create_module

from mabool.utils.metrics import Metrics

tracing_module = create_module("Tracing")


@tracing_module.provides(scope="singleton")
async def metrics() -> Metrics:
    return Metrics()
