"""
DocsQuery - Retrieval Evaluator

Runs retrieval systems against the ground-truth dataset
and calculates standard information-retrieval metrics.
"""

from collections.abc import Callable

from app.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.models import EvaluationExample
from app.evaluation.results import (
    EvaluationResult,
    MetricResult,
)

# A retrieval function accepts a query and returns ranked
# chunk IDs.
RetrieverFunction = Callable[
    [str, int],
    list[str],
]


class RetrievalEvaluator:
    """
    Evaluates one retrieval strategy against ground truth.
    """

    def evaluate(
        self,
        strategy: str,
        examples: list[EvaluationExample],
        retriever: RetrieverFunction,
    ) -> EvaluationResult:
        """
        Evaluate a retrieval strategy.

        Args:
            strategy:
                Human-readable strategy name.

            examples:
                Ground-truth evaluation examples.

            retriever:
                Function that retrieves ranked chunk IDs.

        Returns:
            Structured evaluation result.
        """

        if not examples:
            raise ValueError("Cannot evaluate an empty dataset.")

        ranked_lists: list[list[str]] = []
        relevant_sets: list[set[str]] = []

        recall_1_values = []
        recall_3_values = []
        recall_5_values = []

        precision_1_values = []
        precision_3_values = []
        precision_5_values = []

        ndcg_5_values = []

        for example in examples:
            # Ask the retrieval strategy for enough candidates
            # to calculate all of our initial metrics.
            retrieved_ids = retriever(
                example.query,
                5,
            )

            relevant_ids = set(example.relevant_chunk_ids)

            ranked_lists.append(retrieved_ids)

            relevant_sets.append(relevant_ids)

            recall_1_values.append(
                recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    1,
                )
            )

            recall_3_values.append(
                recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    3,
                )
            )

            recall_5_values.append(
                recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    5,
                )
            )

            precision_1_values.append(
                precision_at_k(
                    retrieved_ids,
                    relevant_ids,
                    1,
                )
            )

            precision_3_values.append(
                precision_at_k(
                    retrieved_ids,
                    relevant_ids,
                    3,
                )
            )

            precision_5_values.append(
                precision_at_k(
                    retrieved_ids,
                    relevant_ids,
                    5,
                )
            )

            ndcg_5_values.append(
                ndcg_at_k(
                    retrieved_ids,
                    relevant_ids,
                    5,
                )
            )

        count = len(examples)

        metrics = MetricResult(
            recall_at_1=(sum(recall_1_values) / count),
            recall_at_3=(sum(recall_3_values) / count),
            recall_at_5=(sum(recall_5_values) / count),
            precision_at_1=(sum(precision_1_values) / count),
            precision_at_3=(sum(precision_3_values) / count),
            precision_at_5=(sum(precision_5_values) / count),
            mrr=mean_reciprocal_rank(
                ranked_lists,
                relevant_sets,
            ),
            ndcg_at_5=(sum(ndcg_5_values) / count),
        )

        return EvaluationResult(
            strategy=strategy,
            num_examples=count,
            metrics=metrics,
        )
