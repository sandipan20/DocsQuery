"""
DocsQuery - Retrieval Score Diagnostics

Analyzes retrieval score distributions for positive and negative
evaluation queries.

This script does NOT change retrieval behavior.

Its purpose is to determine whether retrieval scores can provide
a useful confidence/abstention signal.

Signals analyzed:

    - Top-1 score
    - Mean Top-3 score
    - Mean Top-5 score
    - Top-1 minus Top-2 score gap
    - Top-1 minus Top-5 score gap
"""

from __future__ import annotations

import statistics

from app.config.settings import get_settings
from app.evaluation.dataset import load_evaluation_dataset
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import VectorRetriever


def mean_score(
    scores: list[float],
    k: int,
) -> float:
    """
    Calculate the mean score of the first k results.

    Returns 0.0 when no results are available.
    """

    top_scores = scores[:k]

    if not top_scores:
        return 0.0

    return statistics.mean(top_scores)


def score_gaps(
    scores: list[float],
) -> tuple[float, float]:
    """
    Calculate:

        top-1 minus top-2
        top-1 minus top-5

    Missing ranks are treated as zero.

    This makes the diagnostic safe even when a retriever
    returns fewer than five results.
    """

    if not scores:
        return 0.0, 0.0

    top_1 = scores[0]

    top_2 = scores[1] if len(scores) >= 2 else 0.0
    top_5 = scores[4] if len(scores) >= 5 else 0.0

    return (
        top_1 - top_2,
        top_1 - top_5,
    )


def summarize_signal(
    label: str,
    positive_values: list[float],
    negative_values: list[float],
) -> None:
    """
    Print summary statistics for one diagnostic signal.
    """

    print()
    print(label)

    if positive_values:
        print(
            "  Positive:"
            f" min={min(positive_values):.6f}"
            f" max={max(positive_values):.6f}"
            f" mean={statistics.mean(positive_values):.6f}"
            f" median={statistics.median(positive_values):.6f}"
        )
    else:
        print("  Positive: no data")

    if negative_values:
        print(
            "  Negative:"
            f" min={min(negative_values):.6f}"
            f" max={max(negative_values):.6f}"
            f" mean={statistics.mean(negative_values):.6f}"
            f" median={statistics.median(negative_values):.6f}"
        )
    else:
        print("  Negative: no data")


def summarize_scores(
    name: str,
    positive_top1: list[float],
    negative_top1: list[float],
    positive_top3: list[float],
    negative_top3: list[float],
    positive_top5: list[float],
    negative_top5: list[float],
    positive_gap12: list[float],
    negative_gap12: list[float],
    positive_gap15: list[float],
    negative_gap15: list[float],
) -> None:
    """
    Print all score diagnostics for one retrieval strategy.
    """

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    summarize_signal(
        "Top-1 score",
        positive_top1,
        negative_top1,
    )

    summarize_signal(
        "Mean Top-3 score",
        positive_top3,
        negative_top3,
    )

    summarize_signal(
        "Mean Top-5 score",
        positive_top5,
        negative_top5,
    )

    summarize_signal(
        "Top-1 minus Top-2 gap",
        positive_gap12,
        negative_gap12,
    )

    summarize_signal(
        "Top-1 minus Top-5 gap",
        positive_gap15,
        negative_gap15,
    )


def main() -> None:
    """
    Analyze score distributions for all retrieval strategies.
    """

    dataset = load_evaluation_dataset("data/evaluation/retrieval_dataset.json")

    settings = get_settings()

    # --------------------------------------------------------
    # Build BM25.
    # --------------------------------------------------------

    bm25_storage = BM25Storage(
        settings.bm25_index_path,
    )

    bm25_index = BM25Index(
        storage=bm25_storage,
    )

    bm25_index.load()

    # --------------------------------------------------------
    # Build vector retrieval.
    # --------------------------------------------------------

    vector_retriever = VectorRetriever()

    # --------------------------------------------------------
    # Build hybrid retrieval.
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_index.retriever,
        vector_retriever=vector_retriever,
    )

    # --------------------------------------------------------
    # Build reranker.
    # --------------------------------------------------------

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
    )

    # --------------------------------------------------------
    # Retrieval strategy functions.
    # --------------------------------------------------------

    def bm25_search(query: str):
        return bm25_index.search(
            query=query,
            limit=5,
        )

    def vector_search(query: str):
        return vector_retriever.retrieve(
            query=query,
            limit=5,
        )

    def hybrid_search(query: str):
        return hybrid_retriever.retrieve(
            query=query,
            limit=5,
            candidate_limit=20,
        )

    def reranker_search(query: str):
        candidates = hybrid_retriever.retrieve(
            query=query,
            limit=20,
            candidate_limit=20,
        )

        return reranker.rerank(
            query=query,
            results=candidates,
            top_k=5,
        )

    strategies = {
        "BM25": bm25_search,
        "Vector": vector_search,
        "Hybrid RRF": hybrid_search,
        "Hybrid RRF + Reranker": reranker_search,
    }

    # --------------------------------------------------------
    # Analyze every strategy.
    # --------------------------------------------------------

    for strategy_name, retriever in strategies.items():
        positive_top1: list[float] = []
        negative_top1: list[float] = []

        positive_top3: list[float] = []
        negative_top3: list[float] = []

        positive_top5: list[float] = []
        negative_top5: list[float] = []

        positive_gap12: list[float] = []
        negative_gap12: list[float] = []

        positive_gap15: list[float] = []
        negative_gap15: list[float] = []

        for example in dataset.examples:
            results = retriever(
                example.query,
            )

            if not results:
                continue

            scores = [result.score for result in results]

            top1 = scores[0]

            top3 = mean_score(
                scores,
                3,
            )

            top5 = mean_score(
                scores,
                5,
            )

            gap12, gap15 = score_gaps(
                scores,
            )

            if example.difficulty == "negative":
                negative_top1.append(top1)
                negative_top3.append(top3)
                negative_top5.append(top5)
                negative_gap12.append(gap12)
                negative_gap15.append(gap15)

            else:
                positive_top1.append(top1)
                positive_top3.append(top3)
                positive_top5.append(top5)
                positive_gap12.append(gap12)
                positive_gap15.append(gap15)

        summarize_scores(
            strategy_name,
            positive_top1,
            negative_top1,
            positive_top3,
            negative_top3,
            positive_top5,
            negative_top5,
            positive_gap12,
            negative_gap12,
            positive_gap15,
            negative_gap15,
        )


if __name__ == "__main__":
    main()
