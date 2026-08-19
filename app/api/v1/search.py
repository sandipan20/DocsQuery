"""
DocsQuery - Search API

Exposes hybrid retrieval through HTTP.

Endpoint:

    POST /api/v1/search
"""

from fastapi import APIRouter, Request

from app.api.v1.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.retrieval.models import RetrievalResult

router = APIRouter(
    prefix="/search",
    tags=["search"],
)


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
    request: Request,
    body: SearchRequest,
) -> SearchResponse:
    """
    Search documents using hybrid retrieval.
    """

    # Get the application-wide dependency container.
    container = request.app.state.container

    # Execute hybrid retrieval.
    results = container.retrieval_service.search(
        query=body.query,
        limit=body.limit,
    )

    return SearchResponse(
        query=body.query,
        results=[_to_search_result(result) for result in results],
    )
