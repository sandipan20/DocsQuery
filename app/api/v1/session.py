"""
DocsQuery - Anonymous Session Endpoint
"""

from typing import Any

from fastapi import APIRouter, Request, Response

from app.security.session import get_or_create_session_id

router = APIRouter(
    prefix="/session",
    tags=["session"],
)


@router.get("")
def ensure_session(
    request: Request,
    response: Response,
) -> dict[str, Any]:
    """
    Create an anonymous session if the browser does not have one.
    """

    session_id = get_or_create_session_id(
        request=request,
        response=response,
    )

    return {
        "session_ready": True,
        "session_id": session_id,
    }
