"""
DocsQuery - Application State

Stores expensive application-wide dependencies.

These objects are initialized during application startup
and reused across HTTP requests.
"""

from dataclasses import dataclass

from app.services.retrieval_service import RetrievalService


@dataclass
class AppState:
    """
    Runtime dependencies shared by the FastAPI application.
    """

    retrieval_service: RetrievalService
