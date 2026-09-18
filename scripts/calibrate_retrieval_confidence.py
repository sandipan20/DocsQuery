"""
DocsQuery - Retrieval Confidence Calibration

Sweeps candidate confidence thresholds over the labeled retrieval
benchmark.

This is an analysis tool only. It does not change production
retrieval or RAG behavior.

For each threshold we measure:

    False Accept Rate:
        negative query accepted by the confidence rule.

    False Reject Rate:
        non-negative query rejected by the confidence rule.

    Acceptance Rate:
        all queries accepted by the confidence rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from app.config.settings import get_settings
from app.evaluation.dataset import load_evaluation_dataset
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import VectorRetriever


@dataclass
class ExampleSignal:
    """
    Confidence signals collected for one evaluation query.
    """

    example_id: str
    negative: bool
    top1: float
    mean_top3: float
    mean_top5: float


@dataclass
class ThresholdResult:
    """
    Evaluation result for one confidence threshold.
    """

    threshold: float
    false_accept_rate: float
    false_reject_rate: float
    acceptance_rate: float


def mean_score(
    results: list[RetrievalResult],
    k: int,
) -> float:
    """
    Calculate the mean score of the first k results.
    """

    scores = [result.score for result in results[:k]]

    if not scores:
        return 0.0

    return mean(scores)


def collect_signals() -> dict[str, list[ExampleSignal]]:
    """
    Collect confidence signals for vector and reranker retrieval.
    """

    dataset = load_evaluation_dataset("data/evaluation/retrieval_dataset.json")

    settings = get_settings()

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_storage = BM25Storage(
        settings.bm25_index_path,
    )

    bm25_index = BM25Index(
        storage=bm25_storage,
    )

    bm25_index.load()

    # --------------------------------------------------------
    # Vector retrieval
    # --------------------------------------------------------

    vector_retriever = VectorRetriever()

    # --------------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_index.retriever,
        vector_retriever=vector_retriever,
    )

    # --------------------------------------------------------
    # Cross-encoder reranker
    # --------------------------------------------------------

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
    )

    vector_signals: list[ExampleSignal] = []
    reranker_signals: list[ExampleSignal] = []

    for example in dataset.examples:
        # ----------------------------------------------------
        # Vector scores
        # ----------------------------------------------------

        vector_results = vector_retriever.retrieve(
            query=example.query,
            limit=5,
        )

        if vector_results:
            vector_signals.append(
                ExampleSignal(
                    example_id=example.example_id,
                    negative=example.difficulty == "negative",
                    top1=vector_results[0].score,
                    mean_top3=mean_score(
                        vector_results,
                        3,
                    ),
                    mean_top5=mean_score(
                        vector_results,
                        5,
                    ),
                )
            )

        # ----------------------------------------------------
        # Reranker scores
        # ----------------------------------------------------

        candidates = hybrid_retriever.retrieve(
            query=example.query,
            limit=20,
            candidate_limit=20,
        )

        reranked_results = reranker.rerank(
            query=example.query,
            results=candidates,
            top_k=5,
        )

        if reranked_results:
            reranker_signals.append(
                ExampleSignal(
                    example_id=example.example_id,
                    negative=example.difficulty == "negative",
                    top1=reranked_results[0].score,
                    mean_top3=mean_score(
                        reranked_results,
                        3,
                    ),
                    mean_top5=mean_score(
                        reranked_results,
                        5,
                    ),
                )
            )

    return {
        "Vector Top-1": vector_signals,
        "Reranker Top-1": reranker_signals,
        "Reranker Mean Top-3": reranker_signals,
        "Reranker Mean Top-5": reranker_signals,
    }


def evaluate_threshold(
    signals: list[ExampleSignal],
    threshold: float,
    signal_name: str,
) -> ThresholdResult:
    """
    Evaluate a threshold for one signal.
    """

    negative = [signal for signal in signals if signal.negative]

    positive = [signal for signal in signals if not signal.negative]

    def value(signal: ExampleSignal) -> float:
        if signal_name.endswith("Top-1"):
            return signal.top1

        if signal_name.endswith("Top-3"):
            return signal.mean_top3

        if signal_name.endswith("Top-5"):
            return signal.mean_top5

        raise ValueError(f"Unknown signal: {signal_name}")

    negative_accepts = sum(value(signal) >= threshold for signal in negative)

    positive_rejects = sum(value(signal) < threshold for signal in positive)

    all_accepted = sum(value(signal) >= threshold for signal in signals)

    false_accept_rate = negative_accepts / len(negative) if negative else 0.0

    false_reject_rate = positive_rejects / len(positive) if positive else 0.0

    acceptance_rate = all_accepted / len(signals) if signals else 0.0

    return ThresholdResult(
        threshold=threshold,
        false_accept_rate=false_accept_rate,
        false_reject_rate=false_reject_rate,
        acceptance_rate=acceptance_rate,
    )


def print_calibration(
    name: str,
    signals: list[ExampleSignal],
) -> None:
    """
    Sweep thresholds and print promising operating points.
    """

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    if not signals:
        print("No signals available.")
        return

    values = sorted(
        {
            signal.top1
            if name.endswith("Top-1")
            else signal.mean_top3
            if name.endswith("Top-3")
            else signal.mean_top5
            for signal in signals
        }
    )

    # Include thresholds just below/above observed values.
    thresholds: list[float] = []

    for value in values:
        thresholds.append(value)

    best: list[ThresholdResult] = []

    for threshold in thresholds:
        result = evaluate_threshold(
            signals,
            threshold,
            name,
        )

        # Prefer low false-accept rates while keeping
        # false-reject rates as low as possible.
        if result.false_accept_rate <= 0.10:
            best.append(result)

    best.sort(
        key=lambda result: (
            result.false_reject_rate,
            result.threshold,
        )
    )

    print()
    print("Threshold         False Accept     False Reject     Accepted")
    print("----------------------------------------------------------------")

    for result in best[:10]:
        print(
            f"{result.threshold:10.6f}"
            f"        {result.false_accept_rate:7.3f}"
            f"          {result.false_reject_rate:7.3f}"
            f"        {result.acceptance_rate:7.3f}"
        )

    if not best:
        print("No threshold achieved false-accept rate <= 10%.")


def main() -> None:
    """
    Run confidence calibration.
    """

    signals = collect_signals()

    for name, values in signals.items():
        print_calibration(
            name,
            values,
        )


if __name__ == "__main__":
    main()
