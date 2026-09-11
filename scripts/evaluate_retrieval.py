"""
DocsQuery - Retrieval Benchmark Runner

Runs the retrieval evaluation dataset against multiple
retrieval strategies:

    1. BM25
    2. Vector
    3. Hybrid + RRF
    4. Hybrid + RRF + Cross-Encoder

The goal is to measure whether each retrieval stage actually
improves retrieval quality.

This script does NOT call Gemini. It evaluates retrieval only.

Usage:

    python -m scripts.evaluate_retrieval

    python -m scripts.evaluate_retrieval --method bm25
    python -m scripts.evaluate_retrieval --method vector
    python -m scripts.evaluate_retrieval --method hybrid
    python -m scripts.evaluate_retrieval --method reranker
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from app.config.settings import get_settings
from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.evaluator import RetrievalEvaluator
from app.evaluation.results import EvaluationResult
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_storage import BM25Storage
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import VectorRetriever

# A retrieval strategy accepts:
#
#     query
#     limit
#
# and returns ranked chunk IDs.
RetrieverFunction = Callable[
    [str, int],
    list[str],
]


# Public CLI names mapped to the internal benchmark strategy names.
METHODS = {
    "bm25": "BM25",
    "vector": "Vector",
    "hybrid": "Hybrid RRF",
    "reranker": "Hybrid RRF + Reranker",
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Without --method, all retrieval strategies are evaluated.

    With --method, only the selected strategy is evaluated.
    """

    parser = argparse.ArgumentParser(
        description="Run the DocsQuery retrieval benchmark."
    )

    parser.add_argument(
        "--method",
        choices=list(METHODS),
        default=None,
        help=(
            "Evaluate only one retrieval strategy. "
            "Omit this option to evaluate all strategies."
        ),
    )

    return parser.parse_args()


def create_bm25_retriever() -> BM25Index:
    """
    Load the persistent BM25 corpus and create a searchable
    BM25 index.
    """

    settings = get_settings()

    storage = BM25Storage(settings.bm25_index_path)

    index = BM25Index(storage=storage)

    # Restore the persisted corpus into the in-memory BM25
    # ranking structure.
    index.load()

    return index


def build_strategies() -> dict[
    str,
    RetrieverFunction,
]:
    """
    Construct all retrieval strategies used in the benchmark.

    Returns:
        Mapping from strategy name to retrieval function.
    """

    # --------------------------------------------------------
    # Load shared retrieval components.
    # --------------------------------------------------------

    bm25_index = create_bm25_retriever()

    vector_retriever = VectorRetriever()

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_index.retriever,
        vector_retriever=vector_retriever,
    )

    settings = get_settings()

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
    )

    # --------------------------------------------------------
    # BM25 strategy.
    # --------------------------------------------------------

    def bm25(
        query: str,
        limit: int,
    ) -> list[str]:
        results = bm25_index.search(
            query=query,
            limit=limit,
        )

        return [result.chunk_id for result in results]

    # --------------------------------------------------------
    # Vector strategy.
    # --------------------------------------------------------

    def vector(
        query: str,
        limit: int,
    ) -> list[str]:
        results = vector_retriever.retrieve(
            query=query,
            limit=limit,
        )

        return [result.chunk_id for result in results]

    # --------------------------------------------------------
    # Hybrid RRF strategy.
    # --------------------------------------------------------

    def hybrid(
        query: str,
        limit: int,
    ) -> list[str]:
        results = hybrid_retriever.retrieve(
            query=query,
            limit=limit,
            candidate_limit=max(
                limit * 4,
                20,
            ),
        )

        return [result.chunk_id for result in results]

    # --------------------------------------------------------
    # Hybrid + reranker strategy.
    # --------------------------------------------------------

    def hybrid_reranked(
        query: str,
        limit: int,
    ) -> list[str]:
        candidates = hybrid_retriever.retrieve(
            query=query,
            limit=20,
            candidate_limit=20,
        )

        results = reranker.rerank(
            query=query,
            results=candidates,
            top_k=limit,
        )

        return [result.chunk_id for result in results]

    return {
        "BM25": bm25,
        "Vector": vector,
        "Hybrid RRF": hybrid,
        "Hybrid RRF + Reranker": hybrid_reranked,
    }


def print_result(
    result: EvaluationResult,
) -> None:
    """
    Print one evaluation result in a readable format.
    """

    metrics = result.metrics

    print()
    print("=" * 70)
    print(result.strategy)
    print("=" * 70)

    # --------------------------------------------------------
    # Evaluation population.
    # --------------------------------------------------------

    print(f"Total examples:        {result.num_examples}")
    print(f"Non-negative examples: {result.num_non_negative_examples}")
    print(f"Adversarial examples:  {result.num_adversarial_examples}")
    print(f"Negative examples:     {result.num_negative_examples}")

    print()

    # --------------------------------------------------------
    # Standard retrieval metrics.
    #
    # Negative examples are excluded from these metrics.
    # --------------------------------------------------------

    print("Standard Retrieval Metrics")
    print("-" * 70)

    print(f"Recall@1:      {metrics.recall_at_1:.4f}")

    print(f"Recall@3:      {metrics.recall_at_3:.4f}")

    print(f"Recall@5:      {metrics.recall_at_5:.4f}")

    print(f"Recall@10:     {metrics.recall_at_10:.4f}")

    print(f"Recall@20:     {metrics.recall_at_20:.4f}")

    print(f"Precision@1:   {metrics.precision_at_1:.4f}")

    print(f"Precision@3:   {metrics.precision_at_3:.4f}")

    print(f"Precision@5:   {metrics.precision_at_5:.4f}")

    print(f"MRR:           {metrics.mrr:.4f}")

    print(f"nDCG@5:        {metrics.ndcg_at_5:.4f}")

    print()

    # --------------------------------------------------------
    # Negative-query safety metrics.
    # --------------------------------------------------------

    print("Negative Retrieval Safety")
    print("-" * 70)

    print(f"False Retrieval Rate@1:  {metrics.false_retrieval_rate_at_1:.4f}")

    print(f"False Retrieval Rate@3:  {metrics.false_retrieval_rate_at_3:.4f}")

    print(f"False Retrieval Rate@5:  {metrics.false_retrieval_rate_at_5:.4f}")

    print(f"False Retrieval Rate@10: {metrics.false_retrieval_rate_at_10:.4f}")

    print(f"False Retrieval Rate@20: {metrics.false_retrieval_rate_at_20:.4f}")


def save_results(
    results: list[EvaluationResult],
    output_path: str,
) -> None:
    """
    Save benchmark results as JSON.

    Keeping the result file makes experiments reproducible
    and allows us to compare future versions of DocsQuery.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = [result.model_dump() for result in results]

    path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """
    Run the retrieval benchmark.

    By default all strategies are evaluated.

    When --method is supplied, only the selected strategy
    is evaluated.
    """

    args = parse_args()

    # --------------------------------------------------------
    # Load the ground-truth evaluation dataset.
    # --------------------------------------------------------

    dataset = load_evaluation_dataset("data/evaluation/retrieval_dataset.json")

    print()
    print("DocsQuery Retrieval Benchmark")
    print("=" * 70)
    print(f"Evaluation examples: {len(dataset.examples)}")

    # --------------------------------------------------------
    # Build retrieval strategies.
    # --------------------------------------------------------

    strategies = build_strategies()

    # --------------------------------------------------------
    # Select one strategy when requested.
    # --------------------------------------------------------

    if args.method is not None:
        selected_strategy = METHODS[args.method]

        strategies = {
            selected_strategy: strategies[selected_strategy],
        }

        print(f"Selected method: {selected_strategy}")

    evaluator = RetrievalEvaluator()

    evaluation_results = []

    # --------------------------------------------------------
    # Evaluate selected strategy/strategies using the same
    # ground-truth dataset.
    # --------------------------------------------------------

    for strategy_name, retriever in strategies.items():
        print()
        print(f"Evaluating: {strategy_name}...")

        result = evaluator.evaluate(
            strategy=strategy_name,
            examples=dataset.examples,
            retriever=retriever,
        )

        evaluation_results.append(result)

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    for result in evaluation_results:
        print_result(result)

    # --------------------------------------------------------
    # Save results for later comparison and CI.
    # --------------------------------------------------------

    output_path = "data/evaluation/retrieval_results.json"

    save_results(
        evaluation_results,
        output_path,
    )

    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
