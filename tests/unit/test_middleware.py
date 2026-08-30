"""
Tests for API middleware.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware import (
    request_id_middleware,
)


def create_test_app():
    """
    Create a minimal FastAPI application for middleware tests.
    """

    app = FastAPI()

    app.middleware("http")(request_id_middleware)

    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}

    return app


def test_request_id_is_generated():
    """
    Requests without an ID should receive one.
    """

    client = TestClient(create_test_app())

    response = client.get("/test")

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")

    assert response.headers.get("X-Process-Time")


def test_existing_request_id_is_preserved():
    """
    A client-provided request ID should be preserved.
    """

    client = TestClient(create_test_app())

    response = client.get(
        "/test",
        headers={"X-Request-ID": "test-request-123"},
    )

    assert response.headers["X-Request-ID"] == "test-request-123"
