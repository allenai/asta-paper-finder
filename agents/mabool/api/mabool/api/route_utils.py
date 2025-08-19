from typing import Any

from fastapi.responses import JSONResponse, Response


def create_json_response(content: Any) -> Response:
    headers = {"Cache-Control": "public, max-age=31536000, immutable"}
    return JSONResponse(status_code=200, content=content, headers=headers)
