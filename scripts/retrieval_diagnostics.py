"""
DocsQuery - Retrieval Diagnostics

Produces detailed per-query retrieval diagnostics.

For every query and retrieval method we record:

    - query
    - relevant chunk IDs
    - score
    - source
    - page
    - text preview
    - first relevant rank
    - hit@5
    - hit@10
    - hit@20

This helps identify whether retrieval failures are caused by:

    1. ranking
    2. embeddings
    3. keyword matching
    4. chunking
    5. benchmark annotations
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config.settings import get_settings
from app.evaluation.dataset import load_evaluation_dataset
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.vector_store import QdrantVectorStore

DEFAULT_DATASET = "data/evaluation/retrieval_dataset.json"

DEFAULT_OUTPUT = "data/evaluation/retrieval_diagnostics.json"

TEXT_PREVIEW_LENGTH = 220

# Request enough candidates to distinguish:
#
#   top-5
#   top-10
#   top-20
#
# This lets us determine whether a relevant chunk is missing
# completely or is merely ranked too low.
DIAGNOSTIC_RETRIEVAL_LIMIT = 20


@dataclass(frozen=True)
class RetrievedItem:
    """
    Detailed information about one retrieved result.
    """

    rank: int
    chunk_id: str
    score: float
    source: str
    page_number: int
    text_preview: str
    relevant: bool


@dataclass(frozen=True)
class DiagnosticRecord:
    """
    Diagnostics for one query and one retrieval method.
    """

    example_id: str
    query: str
    method: str

    relevant_chunk_ids: list[str]

    retrieved_items: list[RetrievedItem]

    first_relevant_rank: int | None

    hit_at_5: bool

    hit_at_10: bool

    hit_at_20: bool


def first_relevant_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> int | None:
    """
    Return the 1-based rank of the first relevant chunk.

    Returns:
        None if no relevant chunk was retrieved.
    """

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_ids:
            return rank

    return None


def make_record(
    *,
    example_id: str,
    query: str,
    method: str,
    relevant_ids: list[str],
    results,
) -> DiagnosticRecord:
    """
    Convert retrieval results into detailed diagnostics.
    """

    relevant_set = set(relevant_ids)

    retrieved_items = [
        RetrievedItem(
            rank=rank,
            chunk_id=result.chunk_id,
            score=float(result.score),
            source=result.source,
            page_number=result.page_number,
            text_preview=result.text[:TEXT_PREVIEW_LENGTH].replace(
                "\n",
                " ",
            ),
            relevant=result.chunk_id in relevant_set,
        )
        for rank, result in enumerate(
            results,
            start=1,
        )
    ]

    retrieved_ids = [result.chunk_id for result in results]

    rank = first_relevant_rank(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_set,
    )

    return DiagnosticRecord(
        example_id=example_id,
        query=query,
        method=method,
        relevant_chunk_ids=relevant_ids,
        retrieved_items=retrieved_items,
        first_relevant_rank=rank,
        hit_at_5=(rank is not None and rank <= 5),
        hit_at_10=(rank is not None and rank <= 10),
        hit_at_20=(rank is not None and rank <= 20),
    )


def run_diagnostics(
    *,
    dataset_path: str,
    output_path: str,
) -> list[DiagnosticRecord]:
    """
    Run detailed retrieval diagnostics.
    """

    settings = get_settings()

    # Load the validated evaluation dataset.
    #
    # load_evaluation_dataset() returns an EvaluationDataset
    # object, not a tuple.
    dataset = load_evaluation_dataset(
        dataset_path,
    )

    # The actual evaluation examples are stored in .examples.
    examples = dataset.examples

    # --------------------------------------------------------
    # Create the shared vector retrieval components.
    # --------------------------------------------------------

    embedding_service = EmbeddingService()

    vector_store = QdrantVectorStore()

    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
    )

    # --------------------------------------------------------
    # Load the persistent BM25 index.
    # --------------------------------------------------------

    bm25_storage = BM25Storage(
        settings.bm25_index_path,
    )

    bm25_index = BM25Index(
        storage=bm25_storage,
    )

    bm25_index.load()

    records: list[DiagnosticRecord] = []

    # --------------------------------------------------------
    # Evaluate every query.
    # --------------------------------------------------------

    for example in examples:
        # ----------------------------------------------------
        # BM25 retrieval
        # ----------------------------------------------------

        bm25_results = bm25_index.search(
            query=example.query,
            limit=DIAGNOSTIC_RETRIEVAL_LIMIT,
        )

        records.append(
            make_record(
                example_id=example.example_id,
                query=example.query,
                method="bm25",
                relevant_ids=example.relevant_chunk_ids,
                results=bm25_results,
            )
        )

        # ----------------------------------------------------
        # Vector retrieval
        # ----------------------------------------------------

        vector_results = vector_retriever.retrieve(
            query=example.query,
            limit=DIAGNOSTIC_RETRIEVAL_LIMIT,
        )

        records.append(
            make_record(
                example_id=example.example_id,
                query=example.query,
                method="vector",
                relevant_ids=example.relevant_chunk_ids,
                results=vector_results,
            )
        )

    # --------------------------------------------------------
    # Build the JSON diagnostics payload.
    # --------------------------------------------------------

    payload = {
        "dataset": dataset_path,
        "retrieval_limit": DIAGNOSTIC_RETRIEVAL_LIMIT,
        "methods": [
            "bm25",
            "vector",
        ],
        "records": [asdict(record) for record in records],
    }

    # --------------------------------------------------------
    # Persist the diagnostics report.
    # --------------------------------------------------------

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")

    return records


def print_record(
    record: DiagnosticRecord,
    top_n: int = 10,
) -> None:
    """
    Print the most important retrieval results for one query.
    """

    print()
    print("=" * 100)
    print("Example:", record.example_id)
    print("Method:", record.method)
    print("Query:", record.query)

    print(
        "Relevant:",
        record.relevant_chunk_ids,
    )

    print(
        "First relevant rank:",
        record.first_relevant_rank,
    )

    print(
        "Hit@5:",
        record.hit_at_5,
    )

    print(
        "Hit@10:",
        record.hit_at_10,
    )

    print(
        "Hit@20:",
        record.hit_at_20,
    )

    print()
    print("Top results:")
    print("-" * 100)

    for item in record.retrieved_items[:top_n]:
        marker = "RELEVANT" if item.relevant else ""

        print(
            f"{item.rank:2d}. "
            f"score={item.score:.6f} "
            f"source={item.source} "
            f"page={item.page_number} "
            f"{marker}"
        )

        print(f"    chunk={item.chunk_id}")

        print(f"    text={item.text_preview}")


def print_summary(
    records: list[DiagnosticRecord],
) -> None:
    """
    Print an aggregate summary.

    The summary reports retrieval depth so we can see whether
    failures are true retrieval failures or simply ranking
    failures within the original top-5 window.
    """

    methods = sorted({record.method for record in records})

    print()
    print("=" * 90)
    print("Retrieval Diagnostics Summary")
    print("=" * 90)

    for method in methods:
        method_records = [record for record in records if record.method == method]

        hit_5 = sum(record.hit_at_5 for record in method_records)

        hit_10 = sum(record.hit_at_10 for record in method_records)

        hit_20 = sum(record.hit_at_20 for record in method_records)

        print(
            f"{method:10s} "
            f"Hit@5={hit_5}/{len(method_records)}  "
            f"Hit@10={hit_10}/{len(method_records)}  "
            f"Hit@20={hit_20}/{len(method_records)}"
        )

    print("=" * 90)


def main() -> None:
    """
    CLI entry point.
    """

    records = run_diagnostics(
        dataset_path=DEFAULT_DATASET,
        output_path=DEFAULT_OUTPUT,
    )

    print_summary(records)

    # Print queries that failed to retrieve a relevant chunk
    # within the original top-5 ranking window.
    #
    # Some of these may still succeed at rank 6-20.
    for record in records:
        if not record.hit_at_5:
            print_record(record)

    print()

    print(f"Diagnostic report written to: {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
