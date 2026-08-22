"""
DocsQuery - Application Dependency Container

Creates and owns long-lived application dependencies.

Dependencies are initialized once per application process
and reused across requests.

Current dependencies:

    BM25
    Vector Retriever
    Cross-Encoder Reranker
    Gemini LLM
    Retrieval Service
"""

from app.config.settings import get_settings
from app.generation.citation_validator import (
    CitationValidator,
)
from app.generation.context_builder import (
    ContextBuilder,
)
from app.generation.llm import LLMService
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import VectorRetriever
from app.services.generation_service import (
    GenerationService,
)
from app.services.retrieval_service import RetrievalService


class AppContainer:
    """
    Holds application-wide services.

    The container is created once for each application
    process. Expensive dependencies are therefore reused
    across HTTP requests.
    """

    def __init__(self):
        """
        Create application dependencies.
        """

        settings = get_settings()

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        bm25_storage = BM25Storage(settings.bm25_index_path)

        self.bm25_index = BM25Index(storage=bm25_storage)

        # ----------------------------------------------------
        # Vector retrieval
        # ----------------------------------------------------

        self.vector_retriever = VectorRetriever()

        # ----------------------------------------------------
        # Cross-encoder reranker
        # ----------------------------------------------------

        # The reranker model is loaded once when the
        # application starts.
        self.reranker = CrossEncoderReranker(model_name=settings.reranker_model)

        # ----------------------------------------------------
        # Gemini LLM
        # ----------------------------------------------------

        # The Gemini client is also created once.
        #
        # We do NOT create it inside an HTTP request because
        # that would unnecessarily recreate the client.
        self.llm = LLMService(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            temperature=settings.gemini_temperature,
            max_tokens=settings.gemini_max_tokens,
        )

        # ----------------------------------------------------
        # Complete retrieval service
        # ----------------------------------------------------

        self.retrieval_service = RetrievalService(
            bm25_index=self.bm25_index,
            vector_retriever=self.vector_retriever,
            reranker=self.reranker,
            candidate_limit=20,
            top_k=settings.reranker_top_k,
        )

        # ----------------------------------------------------
        # Application readiness state
        # ----------------------------------------------------

        self.bm25_loaded = False

        self.generation_service = GenerationService(
            llm_service=self.llm,
            context_builder=ContextBuilder(),
            citation_validator=CitationValidator(),
        )

    def load_indexes(self) -> int:
        """
        Load persistent retrieval indexes.

        Returns:
            Number of BM25 chunks loaded.
        """

        # If no persisted BM25 index exists, the application
        # is not ready for retrieval yet.
        if not self.bm25_index.storage.exists():
            return 0

        count = self.bm25_index.load()

        self.bm25_loaded = count > 0

        return count
