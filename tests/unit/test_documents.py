from unittest.mock import MagicMock

import pytest

from app.documents.service import (
    MAX_FILE_SIZE_BYTES,
    DocumentService,
    InvalidDocumentError,
)
from app.ingestion.models import DocumentPage


def create_service():
    bm25_index = MagicMock()
    bm25_index.retriever.chunks = []
    vector_indexer = MagicMock()
    vector_store = MagicMock()
    return DocumentService(bm25_index, vector_indexer, vector_store)


def test_upload_indexes_pages_into_the_requested_workspace(monkeypatch):
    from app.documents import service

    monkeypatch.setattr(
        service,
        "load_pdf",
        lambda _: [
            DocumentPage(page_number=1, text="Useful evidence", source="temp.pdf"),
            DocumentPage(page_number=2, text="More evidence", source="temp.pdf"),
        ],
    )
    document_service = create_service()

    documents = document_service.upload(
        [("notes.pdf", b"%PDF-1.7 test")],
        workspace_id="workspace-a",
    )

    assert documents[0].filename == "notes.pdf"
    assert documents[0].page_count == 2
    assert documents[0].chunk_count == 2
    indexed_chunks = document_service.vector_indexer.index_chunks.call_args.args[0]
    assert {chunk.workspace_id for chunk in indexed_chunks} == {"workspace-a"}
    document_service.bm25_index.build.assert_called_once()


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("", b"%PDF-1.7"),
        ("notes.txt", b"%PDF-1.7"),
        ("notes.pdf", b""),
        ("notes.pdf", b"not a pdf"),
        ("notes.pdf", b"%PDF-1.7" + b"x" * MAX_FILE_SIZE_BYTES),
    ],
)
def test_upload_rejects_invalid_file_inputs(filename, content):
    document_service = create_service()

    with pytest.raises(InvalidDocumentError):
        document_service.upload([(filename, content)], "workspace-a")

    document_service.vector_indexer.index_chunks.assert_not_called()


def test_upload_rejects_pdf_without_extractable_text(monkeypatch):
    from app.documents import service

    monkeypatch.setattr(
        service,
        "load_pdf",
        lambda _: [DocumentPage(page_number=1, text="", source="temp.pdf")],
    )
    document_service = create_service()

    with pytest.raises(InvalidDocumentError, match="no extractable text"):
        document_service.upload([("blank.pdf", b"%PDF-1.7")], "workspace-a")


def test_upload_rejects_malformed_pdf_with_pdf_header():
    document_service = create_service()

    with pytest.raises(InvalidDocumentError, match="could not be read"):
        document_service.upload(
            [("broken.pdf", b"%PDF-1.7 this is not a valid PDF document")],
            "workspace-a",
        )


def test_list_documents_uses_workspace_filtered_vector_payloads():
    document_service = create_service()
    document_service.vector_store.scroll_document_chunks.return_value = [
        {"document_id": "doc-a", "source": "a.pdf", "page_number": 1},
        {"document_id": "doc-a", "source": "a.pdf", "page_number": 2},
    ]

    documents = document_service.list_documents("workspace-a")

    document_service.vector_store.scroll_document_chunks.assert_called_once_with(
        "workspace-a"
    )
    assert documents[0].document_id == "doc-a"
    assert documents[0].page_count == 2
    assert documents[0].chunk_count == 2


def test_list_documents_uses_total_pdf_page_count_metadata():
    document_service = create_service()
    document_service.vector_store.scroll_document_chunks.return_value = [
        {
            "document_id": "doc-a",
            "source": "a.pdf",
            "page_number": 1,
            "document_page_count": 5,
        }
    ]

    documents = document_service.list_documents("workspace-a")

    assert documents[0].page_count == 5


def test_delete_does_not_delete_document_outside_workspace():
    document_service = create_service()
    document_service.vector_store.scroll_document_chunks.return_value = []

    deleted = document_service.delete("workspace-a", "doc-b")

    assert deleted is False
    document_service.vector_store.delete_document.assert_not_called()
    document_service.bm25_index.build.assert_not_called()


def test_delete_filters_qdrant_by_workspace_and_document():
    document_service = create_service()
    document_service.vector_store.scroll_document_chunks.return_value = [
        {"document_id": "doc-a", "source": "a.pdf", "page_number": 1},
    ]

    assert document_service.delete("workspace-a", "doc-a") is True

    document_service.vector_store.delete_document.assert_called_once_with(
        workspace_id="workspace-a",
        document_id="doc-a",
    )
