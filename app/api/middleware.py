"""
DocsQuery - API Middleware

Middleware responsible for request-level concerns such as
request IDs and response headers.
"""

import time
from uuid import uuid4

from fastapi import Request


async def request_id_middleware(
    request: Request,
    call_next,
):
    """
    Attach a request ID and measure request duration.
    """

    request_id = request.headers.get("X-Request-ID")

    if not request_id:
        request_id = str(uuid4())

    request.state.request_id = request_id

    start_time = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    response.headers["X-Request-ID"] = request_id

    response.headers["X-Process-Time"] = f"{duration:.4f}"

    return response
