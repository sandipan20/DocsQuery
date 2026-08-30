"""
Tests for application health checks.
"""

from unittest.mock import MagicMock

from app.services.health_service import (
    HealthService,
)


def test_health_reports_healthy_dependencies():
    """
    Healthy dependencies should be reported as ready.
    """

    container = MagicMock()

    container.bm25_loaded = True

    container.vector_retriever.vector_store.client.get_collections.return_value = (
        MagicMock()
    )

    service = HealthService(container)

    result = service.check()

    assert result == {
        "bm25": True,
        "qdrant": True,
    }


def test_health_reports_qdrant_failure():
    """
    A Qdrant connection failure should make the dependency
    unhealthy rather than crashing the health check.
    """

    container = MagicMock()

    container.bm25_loaded = True

    container.vector_retriever.vector_store.client.get_collections.side_effect = (
        ConnectionError("Qdrant unavailable")
    )

    service = HealthService(container)

    result = service.check()

    assert result == {
        "bm25": True,
        "qdrant": False,
    }
