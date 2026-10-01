import hashlib
import tempfile
from pathlib import Path

from pypdf.errors import PdfReadError

from app.documents.models import DocumentInfo
from app.ingestion.chunker import chunk_pages
from app.ingestion.cleaner import clean_page
from app.ingestion.loader import load_pdf
from app.ingestion.models import DocumentChunk
from app.retrieval.bm25_index import BM25Index
from app.retrieval.indexer import VectorIndexer
from app.retrieval.vector_store import QdrantVectorStore

MAX_UPLOAD_FILES = 10
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


class InvalidDocumentError(ValueError):
    pass


class DocumentService:
    def __init__(
        self,
        bm25_index: BM25Index,
        vector_indexer: VectorIndexer,
        vector_store: QdrantVectorStore,
    ):
        self.bm25_index = bm25_index
        self.vector_indexer = vector_indexer
        self.vector_store = vector_store

    def upload(
        self,
        files: list[tuple[str, bytes]],
        workspace_id: str,
    ) -> list[DocumentInfo]:
        if not workspace_id.strip():
            raise ValueError("workspace_id cannot be empty.")
        if not files or len(files) > MAX_UPLOAD_FILES:
            raise InvalidDocumentError(
                f"Upload between 1 and {MAX_UPLOAD_FILES} PDF files."
            )

        prepared: list[tuple[DocumentInfo, list[DocumentChunk]]] = []
        for filename, content in files:
            prepared.append(self._prepare_file(filename, content, workspace_id))

        all_chunks = [chunk for _, chunks in prepared for chunk in chunks]
        self.vector_indexer.index_chunks(all_chunks)

        existing_chunks = list(self.bm25_index.retriever.chunks)
        replaced_documents = {(workspace_id, info.document_id) for info, _ in prepared}
        retained_chunks = [
            chunk
            for chunk in existing_chunks
            if (chunk.workspace_id, chunk.document_id) not in replaced_documents
        ]
        self.bm25_index.build(
            [*retained_chunks, *all_chunks],
            persist=True,
        )
        return [info for info, _ in prepared]

    def list_documents(self, workspace_id: str) -> list[DocumentInfo]:
        payloads = self.vector_store.scroll_document_chunks(workspace_id)
        documents: dict[str, DocumentInfo] = {}
        for payload in payloads:
            document_id = payload.get("document_id")
            source = payload.get("source")
            page_number = payload.get("page_number")
            document_page_count = payload.get("document_page_count")
            if not document_id or not source or not page_number:
                continue
            if document_id not in documents:
                documents[document_id] = DocumentInfo(
                    document_id=document_id,
                    filename=source,
                    page_count=document_page_count or page_number,
                    chunk_count=1,
                )
            else:
                current = documents[document_id]
                current.page_count = max(
                    current.page_count,
                    document_page_count or page_number,
                )
                current.chunk_count += 1
        return sorted(documents.values(), key=lambda item: item.filename.lower())

    def delete(self, workspace_id: str, document_id: str) -> bool:
        owned_payloads = self.vector_store.scroll_document_chunks(
            workspace_id=workspace_id,
            document_id=document_id,
        )
        if not owned_payloads:
            return False

        self.vector_store.delete_document(
            workspace_id=workspace_id,
            document_id=document_id,
        )
        retained_chunks = [
            chunk
            for chunk in self.bm25_index.retriever.chunks
            if not (
                chunk.workspace_id == workspace_id and chunk.document_id == document_id
            )
        ]
        self.bm25_index.build(retained_chunks, persist=True)
        return True

    @staticmethod
    def _prepare_file(
        filename: str,
        content: bytes,
        workspace_id: str,
    ) -> tuple[DocumentInfo, list[DocumentChunk]]:
        if not filename.strip():
            raise InvalidDocumentError("Filename cannot be empty.")
        if Path(filename).suffix.lower() != ".pdf":
            raise InvalidDocumentError("Only PDF files are supported.")
        if not content:
            raise InvalidDocumentError("The uploaded PDF is empty.")
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise InvalidDocumentError("PDF files must be 25 MB or smaller.")
        if b"%PDF-" not in content[:1024]:
            raise InvalidDocumentError("The uploaded file is not a valid PDF.")

        document_id = hashlib.sha256(content).hexdigest()
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                pages = load_pdf(temporary_file.name)
        except (OSError, PdfReadError, ValueError) as exc:
            raise InvalidDocumentError("The uploaded PDF could not be read.") from exc

        pages = [
            clean_page(page.model_copy(update={"source": Path(filename).name}))
            for page in pages
        ]
        if not pages or not any(page.text for page in pages):
            raise InvalidDocumentError("The PDF contains no extractable text.")

        chunks = chunk_pages(pages, document_id=document_id)
        chunks = [
            chunk.model_copy(update={"workspace_id": workspace_id}) for chunk in chunks
        ]
        return (
            DocumentInfo(
                document_id=document_id,
                filename=Path(filename).name,
                page_count=len(pages),
                chunk_count=len(chunks),
            ),
            chunks,
        )
