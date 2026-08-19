"""
DocsQuery - Application Configuration

This module is responsible for loading and validating
configuration values from environment variables.

The application should access configuration through the
Settings object instead of reading environment variables
throughout the codebase.
"""

from functools import lru_cache

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

    debug: bool = False

    # --------------------------------------------------------
    # LLM configuration
    # --------------------------------------------------------

    llm_api_key: str = ""

    llm_model: str = ""

    # --------------------------------------------------------
    # Qdrant configuration
    # --------------------------------------------------------

    qdrant_url: str = ""

    qdrant_api_key: str = ""

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

    top_k_rerank: int = 5

    # --------------------------------------------------------
    # Pydantic Settings configuration
    # --------------------------------------------------------

    model_config = SettingsConfigDict(
        # Read configuration from the .env file.
        env_file=".env",
        # Ignore environment variables that are not defined
        # as fields in this Settings class.
        extra="ignore",
        # Environment variable names are case-insensitive.
        case_sensitive=False,
    )
    # --------------------------------------------------------
    # Qdrant configuration
    # --------------------------------------------------------

    # Local Qdrant server address.
    qdrant_url: str = "http://localhost:6333"

    # Collection where document chunks will be stored.
    qdrant_collection: str = "docsquery_chunks"

    # --------------------------------------------------------
    # BM25 configuration
    # --------------------------------------------------------

    bm25_index_path: str = "data/index/bm25.json"

    # --------------------------------------------------------
    # Reranker configuration
    # --------------------------------------------------------

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    reranker_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    """
    Return the application configuration.

    @lru_cache means the Settings object is created once
    and reused instead of reading the .env file every time
    get_settings() is called.
    """

    return Settings()
