"""
DocsQuery - API Error Handling

Centralized exception handlers for predictable API responses.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Handle unexpected application exceptions.

    The actual exception is logged internally, while the
    client receives a safe generic error message.
    """

    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    logger.exception(
        "Unhandled exception",
        extra={
            "request_id": request_id,
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": ("An unexpected error occurred."),
                "request_id": request_id,
            }
        },
        headers={
            "X-Request-ID": request_id,
        },
    )
