import logging
import sys
import traceback
from typing import Any

import uvicorn
from ai2i.config import application_config_ctx, config_value
from ai2i.config.config_models import AppConfig
from ai2i.config.loading import load_conf
from ai2i.di import create_app_context, create_managed_app
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from mabool.api.round_v2_routes import router as rounds_v2_routes
from mabool.data_model.config import cfg_schema
from mabool.services.services_deps import services_module
from mabool.utils.logging import initialize_logging
from mabool.utils.paths import project_root


def create_app(di_patched_instances: dict[str, Any] | None = None, **config_overrides: dict[str, Any]) -> FastAPI:
    # start by loading app config
    app_config = load_conf(project_root() / "conf")
    config_settings = app_config.config.merge_dict(config_overrides)

    if di_patched_instances is None:
        di_patched_instances = {}

    app_ctx = create_app_context(services_module)

    # create an app that manages dependency injection scopes (and config)
    app = create_managed_app(app_ctx, config_settings, di_patched_instances, project_root() / "conf")

    with application_config_ctx(AppConfig(config=config_settings, user_facing=app_config.user_facing)):
        logger = initialize_logging(
            app,
            config_value(cfg_schema.log_max_length, default=sys.maxsize),
            config_value(cfg_schema.log_format, default=""),
        )

        setup_cors(app)
        setup_error_handlers(app, logger)
        setup_basic_routes(app)

    # API v2 routes
    app.include_router(rounds_v2_routes)

    logger.info(f"🌊🌊🌊 app created ({config_settings.env=})  🌊🌊🌊")
    return app


def setup_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )


def setup_basic_routes(app: FastAPI) -> None:
    @app.get("/")
    def root() -> Response:
        return RedirectResponse("/docs")

    # This tells the machinery that powers Skiff (Kubernetes) that your application
    # is ready to receive traffic. Returning a non 200 response code will prevent the
    # application from receiving live requests.
    @app.get("/health", status_code=204)
    def health() -> Response:
        return Response(status_code=204)


def setup_error_handlers(app: FastAPI, logger: logging.Logger) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, str):
            return JSONResponse(jsonable_encoder({"error": exc.detail}), status_code=exc.status_code)
        elif isinstance(exc.detail, dict):  # dict should already contain a json with "error" key.
            return JSONResponse(jsonable_encoder(exc.detail), status_code=exc.status_code)
        else:
            return JSONResponse(jsonable_encoder({"error": exc.detail}), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(jsonable_encoder({"error": {"detail": exc.errors()}}), status_code=422)

    @app.exception_handler(Exception)
    async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.warning(traceback.format_exc())
        return JSONResponse(jsonable_encoder({"error": str(exc)}), status_code=500)


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
