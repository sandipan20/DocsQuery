from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import Response as EmptyResponse

from app.documents.models import DocumentInfo
from app.documents.service import InvalidDocumentError
from app.security.session import get_or_create_session_id

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=list[DocumentInfo], status_code=201)
async def upload_documents(
    request: Request,
    response: Response,
    files: list[UploadFile] = File(...),
) -> list[DocumentInfo]:
    workspace_id = get_or_create_session_id(request, response)
    if not files or len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Upload between 1 and 10 PDF files.",
        )

    file_data = []
    for upload in files:
        content = await upload.read(25 * 1024 * 1024 + 1)
        file_data.append((upload.filename or "", content))

    try:
        return request.app.state.container.document_service.upload(
            files=file_data,
            workspace_id=workspace_id,
        )
    except InvalidDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=list[DocumentInfo])
def list_documents(request: Request, response: Response) -> list[DocumentInfo]:
    workspace_id = get_or_create_session_id(request, response)
    return request.app.state.container.document_service.list_documents(workspace_id)


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    request: Request,
    response: Response,
) -> EmptyResponse:
    workspace_id = get_or_create_session_id(request, response)
    deleted = request.app.state.container.document_service.delete(
        workspace_id=workspace_id,
        document_id=document_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return EmptyResponse(status_code=204)
