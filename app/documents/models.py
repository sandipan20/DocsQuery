from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
