"""
DocsQuery - Health Service

Checks whether application dependencies are ready.

This service deliberately does not import AppContainer.
Doing so would create a circular import because AppContainer
creates HealthService.
"""


class HealthService:
    """
    Performs application readiness checks.
    """

    def __init__(self, container):
        """
        Initialize the health service.

        The container argument is intentionally untyped here.

        Why?
        -------
        AppContainer creates HealthService, so importing
        AppContainer into this module would create a circular
        dependency.
        """

        self.container = container

    def check(self) -> dict[str, bool]:
        """
        Return readiness information for application dependencies.
        """

        return {
            "bm25": self.container.bm25_loaded,
            "qdrant": self._check_qdrant(),
        }

    def _check_qdrant(self) -> bool:
        """
        Check whether Qdrant is reachable.

        Returns:
            True if Qdrant responds successfully.
            False if the connection fails.
        """

        try:
            self.container.vector_retriever.vector_store.client.get_collections()

            return True

        except Exception:
            return False
