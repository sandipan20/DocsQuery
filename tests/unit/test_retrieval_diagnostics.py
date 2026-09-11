"""
Unit tests for retrieval diagnostics.
"""

from pathlib import Path

from app.retrieval.models import RetrievalResult
from scripts.retrieval_diagnostics import (
    first_relevant_rank,
    make_record,
    run_diagnostics,
)


def create_result(
    chunk_id: str,
    score: float,
) -> RetrievalResult:
    """
    Create a deterministic retrieval result.
    """

    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=f"Text for {chunk_id}",
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        score=score,
    )


def test_first_relevant_rank_returns_one_based_rank():
    """
    The first relevant result should use a 1-based rank.
    """

    assert (
        first_relevant_rank(
            retrieved_ids=[
                "a",
                "b",
                "c",
            ],
            relevant_ids={"b"},
        )
        == 2
    )


def test_first_relevant_rank_returns_none_when_missing():
    """
    None should be returned when no relevant result exists.
    """

    assert (
        first_relevant_rank(
            retrieved_ids=[
                "a",
                "b",
            ],
            relevant_ids={"c"},
        )
        is None
    )


def test_make_record_marks_relevant_result():
    """
    A relevant result should be correctly marked and should
    count as a hit for all appropriate retrieval cutoffs.
    """

    record = make_record(
        example_id="q001",
        query="example",
        method="vector",
        relevant_ids=["b"],
        results=[
            create_result("a", 0.9),
            create_result("b", 0.8),
        ],
    )

    assert record.first_relevant_rank == 2

    assert record.hit_at_5 is True

    assert record.hit_at_10 is True

    assert record.hit_at_20 is True

    assert record.retrieved_items[1].relevant is True


def test_make_record_marks_miss():
    """
    A missing relevant result should produce no hit at any
    diagnostic cutoff.
    """

    record = make_record(
        example_id="q001",
        query="example",
        method="vector",
        relevant_ids=["c"],
        results=[
            create_result("a", 0.9),
            create_result("b", 0.8),
        ],
    )

    assert record.first_relevant_rank is None

    assert record.hit_at_5 is False

    assert record.hit_at_10 is False

    assert record.hit_at_20 is False


def test_make_record_detects_deeper_retrieval_cutoffs():
    """
    A relevant result beyond rank 5 but within rank 10 should
    be considered a miss at top-5 and a hit at top-10 and top-20.
    """

    results = [
        create_result("chunk-1", 1.0),
        create_result("chunk-2", 0.9),
        create_result("chunk-3", 0.8),
        create_result("chunk-4", 0.7),
        create_result("chunk-5", 0.6),
        create_result("chunk-6", 0.5),
        create_result("chunk-7", 0.4),
        create_result("chunk-8", 0.3),
        create_result("chunk-9", 0.2),
        create_result("chunk-a", 0.1),
    ]

    record = make_record(
        example_id="q001",
        query="example",
        method="vector",
        relevant_ids=["chunk-a"],
        results=results,
    )

    assert record.first_relevant_rank == 10

    assert record.hit_at_5 is False

    assert record.hit_at_10 is True

    assert record.hit_at_20 is True


def test_make_record_detects_top_20_retrieval():
    """
    A relevant result between ranks 11 and 20 should be missed
    at top-5 and top-10 but found at top-20.
    """

    results = [
        create_result(
            f"chunk-{index}",
            1.0 - (index * 0.01),
        )
        for index in range(1, 20)
    ]

    results.append(
        create_result(
            "chunk-a",
            0.01,
        )
    )

    record = make_record(
        example_id="q001",
        query="example",
        method="vector",
        relevant_ids=["chunk-a"],
        results=results,
    )

    assert record.first_relevant_rank == 20

    assert record.hit_at_5 is False

    assert record.hit_at_10 is False

    assert record.hit_at_20 is True


def test_make_record_preserves_score_and_text():
    """
    Retrieval metadata should be preserved in the diagnostic
    record.
    """

    record = make_record(
        example_id="q001",
        query="example",
        method="bm25",
        relevant_ids=["a"],
        results=[
            create_result(
                "a",
                12.345,
            ),
        ],
    )

    item = record.retrieved_items[0]

    assert item.score == 12.345

    assert item.source == "test.pdf"

    assert item.page_number == 1

    assert "Text for a" in item.text_preview


def test_run_diagnostics_uses_dataset_examples(
    tmp_path,
    monkeypatch,
):
    """
    Verify that the diagnostic pipeline uses the validated
    dataset examples and writes a report.
    """

    dataset_path = tmp_path / "dataset.json"

    output_path = tmp_path / "diagnostics.json"

    dataset_path.write_text(
        """
        {
          "version": "1.0.0",
          "examples": [
            {
              "example_id": "q001",
              "query": "example",
              "query_type": "easy",
              "difficulty": "easy",
              "relevant_chunk_ids": ["chunk-a"]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    class FakeSettings:
        """
        Minimal settings object required by the diagnostics
        pipeline.
        """

        bm25_index_path = "unused.json"

        top_k_bm25 = 5

        top_k_dense = 5

    class FakeBM25Storage:
        """
        Fake BM25 storage used to isolate this test from disk.
        """

        def __init__(
            self,
            path,
        ):
            self.path = path

    class FakeBM25Index:
        """
        Fake BM25 index returning one deterministic result.
        """

        def __init__(
            self,
            storage,
        ):
            self.storage = storage

        def load(self):
            return 1

        def search(
            self,
            query,
            limit,
        ):
            return [
                create_result(
                    "chunk-a",
                    1.0,
                )
            ]

    class FakeVectorRetriever:
        """
        Fake vector retriever returning one deterministic result.
        """

        def __init__(
            self,
            vector_store,
            embedding_service,
        ):
            pass

        def retrieve(
            self,
            query,
            limit,
        ):
            return [
                create_result(
                    "chunk-a",
                    0.9,
                )
            ]

    # Replace real dependencies with deterministic test doubles.
    monkeypatch.setattr(
        "scripts.retrieval_diagnostics.get_settings",
        lambda: FakeSettings(),
    )

    monkeypatch.setattr(
        "scripts.retrieval_diagnostics.BM25Storage",
        FakeBM25Storage,
    )

    monkeypatch.setattr(
        "scripts.retrieval_diagnostics.BM25Index",
        FakeBM25Index,
    )

    monkeypatch.setattr(
        "scripts.retrieval_diagnostics.VectorRetriever",
        FakeVectorRetriever,
    )

    monkeypatch.setattr(
        "scripts.retrieval_diagnostics.QdrantVectorStore",
        lambda: object(),
    )

    monkeypatch.setattr(
        "scripts.retrieval_diagnostics.EmbeddingService",
        lambda: object(),
    )

    records = run_diagnostics(
        dataset_path=str(dataset_path),
        output_path=str(output_path),
    )

    # One BM25 record and one vector record.
    assert len(records) == 2

    assert output_path.exists()

    text = output_path.read_text(
        encoding="utf-8",
    )

    assert "chunk-a" in text

    assert "0.9" in text


def test_source_file_path_object_is_supported(
    tmp_path: Path,
):
    """
    Basic sanity check for temporary paths used by the tests.
    """

    assert tmp_path.exists()
