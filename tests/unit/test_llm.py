"""
Unit tests for the Gemini LLM service.
"""

from unittest.mock import MagicMock

import pytest
from google.genai import errors

from app.generation.llm import (
    LLMService,
    LLMServiceError,
    LLMServiceUnavailableError,
)


def create_service():
    """
    Create an LLM service with a fake Gemini client.

    No real API request is made.
    """

    service = object.__new__(LLMService)

    service.model = "test-model"
    service.temperature = 0.0
    service.max_tokens = 1000

    service.client = MagicMock()

    return service


def test_empty_query_is_rejected():
    """
    Empty queries should not be sent to Gemini.
    """

    service = create_service()

    with pytest.raises(ValueError):
        service.generate(
            "",
            "Evidence",
        )


def test_empty_context_is_rejected():
    """
    Gemini should never be called without evidence.
    """

    service = create_service()

    with pytest.raises(ValueError):
        service.generate(
            "What is Python?",
            "",
        )


def test_generate_returns_gemini_response():
    """
    Verify that Gemini response text is returned.
    """

    service = create_service()

    response = MagicMock()

    response.text = "Python is a programming language. [C1]"

    service.client.models.generate_content.return_value = response

    answer = service.generate(
        query="What is Python?",
        context=("[C1]\nPython is a programming language."),
    )

    assert answer == ("Python is a programming language. [C1]")

    service.client.models.generate_content.assert_called_once()


def test_empty_gemini_response_raises_error():
    """
    Empty Gemini responses should fail clearly.
    """

    service = create_service()

    response = MagicMock()

    response.text = ""

    service.client.models.generate_content.return_value = response

    with pytest.raises(RuntimeError):
        service.generate(
            query="What is Python?",
            context="[C1] Python is a language.",
        )


def test_gemini_server_error_becomes_unavailable_error():
    """
    Temporary Gemini server failures should become a
    provider-independent unavailable exception.
    """

    service = create_service()

    service.client.models.generate_content.side_effect = errors.ServerError(
        503,
        {
            "error": {
                "code": 503,
                "message": "Service unavailable",
                "status": "UNAVAILABLE",
            }
        },
    )

    with pytest.raises(LLMServiceUnavailableError):
        service.generate(
            query="What is Python?",
            context="[C1] Python is a programming language.",
        )


def test_gemini_api_error_becomes_llm_service_error():
    """
    Other Gemini API failures should become a generic
    application-level LLM exception.
    """

    service = create_service()

    service.client.models.generate_content.side_effect = errors.APIError(
        400,
        {
            "error": {
                "code": 400,
                "message": "Invalid request",
                "status": "INVALID_ARGUMENT",
            }
        },
    )

    with pytest.raises(LLMServiceError):
        service.generate(
            query="What is Python?",
            context="[C1] Python is a programming language.",
        )
