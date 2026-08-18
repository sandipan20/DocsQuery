"""
DocsQuery - Search API

HTTP endpoint for document retrieval.

The route itself should only handle HTTP concerns.
The actual retrieval logic belongs to RetrievalService.
"""

from fastapi import APIRouter, Depends, Request

from app.api.v1.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.retrieval.models import RetrievalResult
from app.services.retrieval_service import RetrievalService

router = APIRouter(
    prefix="/search",
    tags=["search"],
)


def get_retrieval_service(
    request: Request,
) -> RetrievalService:
    """
    Return the application-wide retrieval service.

    The service is created during application startup and
    reused across requests.
    """

    return request.app.state.docsquery.retrieval_service


def _to_search_result(
    result: RetrievalResult,
) -> SearchResult:
    """
    Convert an internal RetrievalResult into the public
    API response model.
    """

    return SearchResult(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        text=result.text,
        source=result.source,
        page_number=result.page_number,
        chunk_index=result.chunk_index,
        score=result.score,
    )


@router.post(
    "",
    response_model=SearchResponse,
)
def search(
    request: SearchRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> SearchResponse:
    """
    Search documents using the retrieval service.

    FastAPI injects the RetrievalService dependency.

    This makes the endpoint easy to unit test without requiring
    Qdrant or any other external service.
    """

    results = service.search(
        query=request.query,
        limit=request.limit,
    )

    return SearchResponse(
        query=request.query,
        results=[_to_search_result(result) for result in results],
    )
