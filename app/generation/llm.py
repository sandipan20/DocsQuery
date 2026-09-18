"""
DocsQuery - Gemini LLM Service

Provides a simple interface around Google's Gemini API.

Architecture:

    Retrieved Evidence
          ↓
    Context Builder
          ↓
      LLMService
          ↓
      Gemini API
          ↓
    Grounded Answer
"""

from google import genai
from google.genai import errors, types

from app.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)


class LLMServiceError(RuntimeError):
    """
    Base exception for expected Gemini service failures.
    """


class LLMServiceUnavailableError(LLMServiceError):
    """
    Raised when Gemini is temporarily unavailable.
    """


class LLMService:
    """
    Generates grounded answers using Google's Gemini API.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ):
        """
        Initialize the Gemini client.

        The client is created once and reused rather than
        being created for every user request.
        """

        if not api_key:
            raise ValueError("Gemini API key is required.")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Create the Gemini client once.
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Generate an answer using only the supplied evidence.

        Args:
            query:
                User's question.

            context:
                Citation-aware retrieved evidence.

        Returns:
            Generated answer.

        Raises:
            LLMServiceUnavailableError:
                If Gemini is temporarily unavailable.

            LLMServiceError:
                If Gemini returns an unexpected API failure.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not context.strip():
            raise ValueError("Context cannot be empty.")

        # Send the question and retrieved evidence to Gemini.
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=build_user_prompt(
                    query=query,
                    context=context,
                ),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )

        except errors.ServerError as exc:
            # Gemini currently reports temporary service-side
            # failures such as HTTP 503 through ServerError.
            raise LLMServiceUnavailableError(
                "The language model is temporarily unavailable. Please try again later."
            ) from exc

        except errors.APIError as exc:
            # Convert other Gemini API failures into an
            # application-specific exception instead of
            # leaking provider exceptions through the API.
            raise LLMServiceError(
                "The language model service failed to generate a response."
            ) from exc

        # Gemini should always return text for this use case.
        if not response.text:
            raise LLMServiceError("Gemini returned an empty response.")

        return response.text.strip()
