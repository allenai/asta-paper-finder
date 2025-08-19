from __future__ import annotations

import logging

from ai2i.di import DI, RequestAndBody, builtin_deps, create_module
from fastapi import Request

from mabool.data_model.rounds import RoundContext, RoundRequest

logger = logging.getLogger(__name__)


context_module = create_module("Contexts")


@context_module.provides(scope="request")
async def request_context(
    request_and_body: RequestAndBody = DI.requires(builtin_deps.request),
) -> RoundContext | None:
    return await _extract_round_context(request_and_body.request, request_and_body)


@context_module.provides(scope="round")
async def round_context(
    first_round_request_and_body: RequestAndBody = DI.requires(builtin_deps.request),
) -> RoundContext | None:
    first_round_request = first_round_request_and_body.request
    return await _extract_round_context(first_round_request, first_round_request_and_body)


async def _extract_round_context(request: Request, request_and_body: RequestAndBody) -> RoundContext | None:
    try:
        match (request.method, request.url.path):
            case ("POST", "/api/2/rounds"):
                json_body = request_and_body.json_body
                round_request = RoundRequest(**json_body)
                return RoundContext(
                    inserted_before=round_request.inserted_before,
                )
            case _:
                return None
    except Exception as e:
        logger.error("Failed to extract round context from request", exc_info=e)
        return None
