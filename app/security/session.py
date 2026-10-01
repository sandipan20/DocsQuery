"""
DocsQuery - Anonymous Session Security

Provides an anonymous, server-generated session identifier.

The session ID is stored in an HttpOnly cookie so frontend
JavaScript cannot read or modify it directly.
"""

import secrets

from fastapi import Request, Response

from app.config.settings import get_settings

SESSION_COOKIE_NAME = "docsquery_session"

# Anonymous sessions live for 24 hours from their last creation.
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24

# 256 bits of randomness encoded as URL-safe text.
SESSION_TOKEN_BYTES = 32


def create_session_id() -> str:
    """
    Create a cryptographically random anonymous session ID.
    """

    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def get_or_create_session_id(
    request: Request,
    response: Response,
    secure_cookie: bool | None = None,
) -> str:
    """
    Return the existing anonymous session ID or create one.

    The session ID is read from the HttpOnly cookie.

    Args:
        request:
            Incoming HTTP request.

        response:
            Response used to set a new cookie when necessary.

        secure_cookie:
            Whether the cookie should include the Secure flag.
            This should be True for HTTPS production deployments.
    """

    header_session_id = (
        request.headers.get("x-session-id")
        or request.headers.get("x-workspace-id")
    )
    existing_session_id = header_session_id or request.cookies.get(
        SESSION_COOKIE_NAME,
    )

    if existing_session_id:
        response.headers["x-session-id"] = existing_session_id
        return existing_session_id

    session_id = create_session_id()
    if secure_cookie is None:
        secure_cookie = get_settings().app_env == "production"

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        path="/",
    )
    response.headers["x-session-id"] = session_id

    return session_id
