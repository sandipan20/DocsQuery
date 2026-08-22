"""
Small manual test for the Gemini API.

This script makes one real API request.

Do NOT run this from CI.
"""

from app.config.settings import get_settings
from app.generation.llm import LLMService


def main():
    """
    Make one test request to Gemini.
    """

    settings = get_settings()

    service = LLMService(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        temperature=settings.gemini_temperature,
        max_tokens=settings.gemini_max_tokens,
    )

    answer = service.generate(
        query="What is Python?",
        context=(
            "[C1]\n"
            "Python is a high-level, general-purpose "
            "programming language."
        ),
    )

    print("\nGemini response:")
    print(answer)


if __name__ == "__main__":
    main()