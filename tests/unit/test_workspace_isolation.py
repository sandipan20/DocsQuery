from app.ingestion.models import DocumentChunk
from app.retrieval.bm25_retriever import BM25Retriever


def create_chunk(
    chunk_id: str,
    workspace_id: str,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        workspace_id=workspace_id,
        text=text,
        source=f"{chunk_id}.pdf",
        page_number=1,
        chunk_index=0,
    )


def test_bm25_returns_only_requested_workspace():
    retriever = BM25Retriever()

    retriever.index(
        [
            create_chunk(
                "a",
                "workspace-a",
                "Python programming language",
            ),
            create_chunk(
                "b",
                "workspace-b",
                "Python programming language",
            ),
        ]
    )

    results = retriever.retrieve(
        query="Python programming",
        limit=10,
        workspace_id="workspace-a",
    )

    assert len(results) == 1
    assert results[0].chunk_id == "a"
    assert results[0].workspace_id == "workspace-a"


def test_bm25_does_not_return_unknown_workspace():
    retriever = BM25Retriever()

    retriever.index(
        [
            create_chunk(
                "a",
                "workspace-a",
                "Python programming language",
            ),
        ]
    )

    results = retriever.retrieve(
        query="Python programming",
        limit=10,
        workspace_id="workspace-b",
    )

    assert results == []


def test_other_workspaces_do_not_change_scoped_bm25_scores():
    user_chunks = [
        create_chunk("a", "workspace-a", "alpha beta gamma"),
        create_chunk("b", "workspace-a", "alpha beta delta"),
    ]
    isolated = BM25Retriever()
    isolated.index(user_chunks)
    shared = BM25Retriever()
    shared.index(
        [
            *user_chunks,
            create_chunk(
                "secret",
                "workspace-b",
                "alpha alpha alpha alpha alpha alpha",
            ),
        ]
    )

    isolated_results = isolated.retrieve(
        "alpha beta",
        limit=10,
        workspace_id="workspace-a",
    )
    shared_results = shared.retrieve(
        "alpha beta",
        limit=10,
        workspace_id="workspace-a",
    )

    assert [(item.chunk_id, item.score) for item in shared_results] == [
        (item.chunk_id, item.score) for item in isolated_results
    ]
