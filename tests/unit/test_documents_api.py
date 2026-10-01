from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.documents.models import DocumentInfo
from app.main import create_app


def test_document_routes_use_only_the_session_cookie_workspace():
    app = create_app()
    document = DocumentInfo(
        document_id="doc-a",
        filename="notes.pdf",
        page_count=2,
        chunk_count=3,
    )

    with TestClient(app) as client:
        service = MagicMock()
        service.upload.return_value = [document]
        service.list_documents.side_effect = lambda workspace_id: (
            [document] if workspace_id == "session-a" else []
        )
        service.delete.side_effect = lambda workspace_id, document_id: (
            workspace_id == "session-a" and document_id == "doc-a"
        )
        app.state.container.document_service = service
        client.cookies.set("docsquery_session", "session-a")

        uploaded = client.post(
            "/api/v1/documents",
            data={"workspace_id": "session-b"},
            files=[
                ("files", ("notes.pdf", b"%PDF-1.7 sample", "application/pdf")),
                ("files", ("extra.pdf", b"%PDF-1.7 extra", "application/pdf")),
            ],
        )
        listed = client.get("/api/v1/documents")
        deleted = client.delete("/api/v1/documents/doc-a")

        client.cookies.set("docsquery_session", "session-b")
        other_session_list = client.get("/api/v1/documents")
        cross_session_delete = client.delete("/api/v1/documents/doc-a")

    assert uploaded.status_code == 201
    service.upload.assert_called_once_with(
        files=[
            ("notes.pdf", b"%PDF-1.7 sample"),
            ("extra.pdf", b"%PDF-1.7 extra"),
        ],
        workspace_id="session-a",
    )
    assert listed.json()[0]["document_id"] == "doc-a"
    assert deleted.status_code == 204
    assert other_session_list.json() == []
    assert cross_session_delete.status_code == 404
    service.delete.assert_called_with(
        workspace_id="session-b",
        document_id="doc-a",
    )
