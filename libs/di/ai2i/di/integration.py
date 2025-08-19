from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable, cast

from fastapi import Depends, FastAPI, Request
from fastapi.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from ai2i.config import ConfigSettings, application_config_ctx, load_user_facing
from ai2i.config.config_models import AppConfig
from ai2i.di.app_context import ApplicationContext
from ai2i.di.interface.models import RequestAndBody
from ai2i.di.managed_env import ManagedEnv


def create_managed_app(
    app_context: ApplicationContext, app_config: ConfigSettings, di_patched_instances: dict[str, Any], conf_path: Path
) -> FastAPI:
    setup_fastapi_di = partial(
        _setup_singleton_scope,
        app_config=app_config,
        app_context=app_context,
        di_patched_instances=di_patched_instances,
        conf_path=conf_path,
    )
    if app_config.show_api_swagger:
        app = FastAPI(lifespan=setup_fastapi_di)
    else:
        app = FastAPI(lifespan=setup_fastapi_di, docs_url=None, redoc_url=None, openapi_url=None)

    # activate DI request scope for http requests
    @app.middleware("http")
    async def _request_scope_in_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        app_config = get_application_config(request.app.state)
        with application_config_ctx(AppConfig(config=app_config, user_facing=load_user_facing(conf_path))):
            singleton_env = get_singleton_env(request.app.state)
            if singleton_env is not None:
                app_context.create_fresh_scopes_context(patched_instances=di_patched_instances)
                app_context.scopes.singleton.replace_env(singleton_env)

            try:
                body = await request.json()
            except Exception:
                body = {}

            await app_context.scopes.request.open_scope(RequestAndBody(request, body))
            try:
                return await call_next(request)
            finally:
                await app_context.scopes.request.close_scope()

    # activate singletone scope for websocket connections
    app.add_middleware(
        WebSocketSingletonScopeMiddleware,
        app_context=app_context,
        di_patched_instances=di_patched_instances,
        conf_path=conf_path,
    )
    return app


class WebSocketSingletonScopeMiddleware:
    _app: ASGIApp
    _app_context: ApplicationContext
    _di_patched_instances: dict[str, Any]
    _conf_path: Path

    def __init__(
        self, app: ASGIApp, app_context: ApplicationContext, di_patched_instances: dict[str, Any], conf_path: Path
    ) -> None:
        self._app = app
        self._app_context = app_context
        self._di_patched_instances = di_patched_instances
        self._conf_path = conf_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            singleton_env = get_singleton_env(scope["app"].state)
            if singleton_env is not None:
                app_config = get_application_config(scope["app"].state)
                with application_config_ctx(
                    AppConfig(config=app_config, user_facing=load_user_facing(self._conf_path))
                ):
                    self._app_context.create_fresh_scopes_context(patched_instances=self._di_patched_instances)
                    self._app_context.scopes.singleton.replace_env(singleton_env)
                    await self._app(scope, receive, send)
            else:
                await self._app(scope, receive, send)
        else:
            await self._app(scope, receive, send)


@asynccontextmanager
async def _setup_singleton_scope(
    app: FastAPI,
    app_context: ApplicationContext,
    app_config: ConfigSettings,
    di_patched_instances: dict[str, Any],
    conf_path: Path,
) -> AsyncGenerator[dict, None]:
    # NOTE: config scope must come first as factories of the singleton scope will need the config available
    with application_config_ctx(AppConfig(config=app_config, user_facing=load_user_facing(conf_path))):
        async with app_context.scopes.singleton.managed_scope({**di_patched_instances}) as singleton_env:
            app_state = _get_app_state(app)
            app_state.application_config = app_config
            app_state.singleton_env = singleton_env
            yield {}


class AppState:
    application_config: ConfigSettings
    singleton_env: ManagedEnv | None


def _get_app_state(app: FastAPI) -> AppState:
    return cast(AppState, app.state)


def get_application_config(app_state: AppState = Depends(_get_app_state)) -> ConfigSettings:
    return app_state.application_config


def get_singleton_env(app_state: AppState = Depends(_get_app_state)) -> ManagedEnv | None:
    return app_state.singleton_env
