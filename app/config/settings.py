"""
DocsQuery - Application Configuration

This module is responsible for loading and validating
configuration values from environment variables.

The application should access configuration through the
Settings object instead of reading environment variables
throughout the codebase.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration object for DocsQuery.

    Values are loaded from:
    1. Environment variables
    2. The local .env file

    Environment variables take priority over .env values.
    """

    # --------------------------------------------------------
    # Application settings
    # --------------------------------------------------------

    app_name: str = "DocsQuery"
    app_version: str = "0.1.0"
    app_env: Literal["development", "production"] = "development"
    debug: bool = False

    # --------------------------------------------------------
    # Gemini configuration
    # --------------------------------------------------------

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_temperature: float = 0.0
    gemini_max_tokens: int = 1000

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --------------------------------------------------------
    # Retrieval configuration
    # --------------------------------------------------------

    top_k_dense: int = 20
    top_k_bm25: int = 20
    reranker_top_k: int = 5

    # Minimum cosine similarity of the strongest vector result
    # required before the query is considered related to the corpus.
    vector_confidence_threshold: float = 0.54

    # --------------------------------------------------------
    # Qdrant configuration
    # --------------------------------------------------------

    # Local development:
    #     http://localhost:6333
    #
    # Docker Compose overrides this with:
    #     http://qdrant:6333
    qdrant_url: str = "http://localhost:6333"

    qdrant_collection: str = "docsquery_chunks"
    qdrant_api_key: str = ""

    # --------------------------------------------------------
    # BM25 configuration
    # --------------------------------------------------------

    bm25_index_path: str = "data/index/bm25.json"

    # --------------------------------------------------------
    # API configuration
    # --------------------------------------------------------

    api_cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    # --------------------------------------------------------
    # Pydantic Settings configuration
    # --------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return the application configuration.

    @lru_cache means the Settings object is created once
    and reused instead of reading the .env file every time
    get_settings() is called.
    """

    return Settings()
