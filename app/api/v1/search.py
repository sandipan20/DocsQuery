"""
DocsQuery - Search API

Exposes hybrid retrieval through HTTP.

Endpoint:

    POST /api/v1/search
"""

from fastapi import APIRouter, Request, Response

from app.api.v1.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.retrieval.models import RetrievalResult
from app.security.session import get_or_create_session_id

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
    response: Response,
    body: SearchRequest,
) -> SearchResponse:
    """
    Search documents using hybrid retrieval.
    """

    # Get the application-wide dependency container.
    container = request.app.state.container
    workspace_id = get_or_create_session_id(
        request=request,
        response=response,
    )
    if body.document_ids is not None:
        owned_ids = {
            document.document_id
            for document in container.document_service.list_documents(workspace_id)
        }
        if not set(body.document_ids).issubset(owned_ids):
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Document not found.")

    # Execute hybrid retrieval.
    results = container.retrieval_service.search(
        query=body.query,
        limit=body.limit,
        workspace_id=workspace_id,
        document_ids=body.document_ids,
    )

    return SearchResponse(
        query=body.query,
        results=[_to_search_result(result) for result in results],
    )
