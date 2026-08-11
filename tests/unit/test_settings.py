"""
Tests for the DocsQuery configuration system.

These tests verify that our application can correctly
load configuration from the environment/.env file.
"""

from app.config.settings import get_settings


def test_settings_load_successfully():
    """
    Verify that the Settings object can be created.

    If configuration loading is broken, this test will fail.
    """

    settings = get_settings()

    # Verify the application name.
    assert settings.app_name == "DocsQuery"

    # Verify the configured application version.
    assert settings.app_version == "0.1.0"


def test_retrieval_configuration():
    """
    Verify that retrieval parameters are loaded correctly.
    """

    settings = get_settings()

    # Dense retrieval should return 20 candidates.
    assert settings.top_k_dense == 20

    # BM25 should return 20 candidates.
    assert settings.top_k_bm25 == 20

    # Reranking should keep the top 5 candidates.
    assert settings.top_k_rerank == 5


def test_model_configuration():
    """
    Verify that the embedding and reranker models
    are configured correctly.
    """

    settings = get_settings()

    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
