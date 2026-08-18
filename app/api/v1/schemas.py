"""
DocsQuery - API Schemas

Defines request and response models used by the HTTP API.

Keeping API schemas separate from internal retrieval models
prevents our HTTP contract from becoming tightly coupled to
the retrieval implementation.
"""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """
    Request body for document search.
    """

    # Natural-language query from the user.
    query: str = Field(
        min_length=1,
        max_length=1000,
    )

    # Number of results requested from the API.
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
    )


class SearchResult(BaseModel):
    """
    One search result returned to an API client.
    """

    chunk_id: str
    document_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    score: float


class SearchResponse(BaseModel):
    """
    Response returned by the search endpoint.
    """

    query: str

    results: list[SearchResult]
