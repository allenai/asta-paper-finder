import logging
import os
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

import json_logging  # type: ignore
from fastapi import FastAPI, Request, Response

correlation_id_context: ContextVar[str] = ContextVar("correlation_id", default="-")


def initialize_logging(
    app: FastAPI,
    max_length: int,
    log_format: str,
) -> logging.Logger:
    @app.middleware("http")
    async def set_correlation_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        correlation_id = str(json_logging.CORRELATION_ID_GENERATOR())
        correlation_id_context.set(correlation_id)
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        return response

    class CloudRunJSONLog(json_logging.JSONLogWebFormatter):
        service_name = f"mabool-{os.environ.get('APP_CONFIG_ENV')}-{os.getenv('DATA_DOMAIN', 'cs')}"
        version = os.environ.get("COMMIT_SHA", "unknown")

        def format(self, record: logging.LogRecord) -> Any:
            original_message = record.getMessage()
            if len(original_message) > max_length:
                record.msg = original_message[:max_length] + "..."
            return super().format(record)

        def _format_log_object(self, record: logging.LogRecord, request_util: json_logging.RequestAdapter) -> Any:
            json_log_object = super(CloudRunJSONLog, self)._format_log_object(record, request_util)
            # Replace the name of key 'level' to 'severity' to match google cloud logging.
            json_log_object["severity"] = json_log_object.pop("level")
            json_log_object["correlation_id"] = correlation_id_context.get()
            json_log_object["message"] = json_log_object.pop("msg")

            if record.exc_info and record.levelno >= logging.ERROR:
                # Format for Google Error Reporting.
                json_log_object["@type"] = (
                    "type.googleapis.com/google.devtools.clouderrorreporting.v1beta1.ReportedErrorEvent"
                )
                json_log_object["serviceContext"] = {
                    "service": self.service_name,
                    "version": self.version,
                }
                json_log_object["stack_trace"] = self.formatException(record.exc_info)
            return json_log_object

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    if log_format == "google:json":
        json_logging.init_fastapi(enable_json=True, custom_formatter=CloudRunJSONLog)
        json_logging.init_request_instrument(app)
        json_logging.config_root_logger()
    return logger
