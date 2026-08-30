"""
DocsQuery - Request Context

Small helpers for accessing request-level information.

Keeping this logic in one module prevents different parts
of the application from accessing request state differently.
"""

from fastapi import Request


def get_request_id(request: Request) -> str:
    """
    Return the request ID assigned by the middleware.

    Args:
        request:
            Current FastAPI request.

    Returns:
        Request ID, or "unknown" if it was not assigned.
    """

    return getattr(
        request.state,
        "request_id",
        "unknown",
    )
