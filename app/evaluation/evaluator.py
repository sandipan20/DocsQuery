"""
DocsQuery - Retrieval Evaluation

Evaluates retrieval performance against labeled evaluation examples.

Positive and adversarial examples use:
    Recall@K
    Precision@K
    MRR@5
    nDCG@5

Negative examples use:
    False Retrieval Rate@K
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from app.evaluation.metrics import (
    mean_false_retrieval_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.models import EvaluationDataset
from app.evaluation.results import EvaluationResult, MetricResult


class RetrievalEvaluator:
    """
    Evaluate retrieval performance against labeled examples.

    The retriever may be supplied during construction or directly to
    evaluate(). The evaluate() method also supports both an
    EvaluationDataset and a raw examples list.

    Retrieval results may be either:
        - strings containing chunk IDs
        - objects with a chunk_id attribute
    """

    def __init__(
        self,
        retriever: Callable[[str, int], Sequence] | None = None,
    ) -> None:
        self.retriever = retriever

    @staticmethod
    def _extract_chunk_ids(
        results: Sequence[Any],
    ) -> list[str]:
        """
        Extract chunk IDs from retrieval results.

        Supports both raw chunk-ID strings and retrieval result objects
        exposing a chunk_id attribute.
        """

        chunk_ids: list[str] = []

        for result in results:
            if isinstance(result, str):
                chunk_ids.append(result)
                continue

            chunk_id = getattr(
                result,
                "chunk_id",
                None,
            )

            if chunk_id:
                chunk_ids.append(chunk_id)

        return chunk_ids

    def evaluate(
        self,
        strategy: str,
        examples: Sequence | None = None,
        retriever: Callable[[str, int], Sequence] | None = None,
        dataset: EvaluationDataset | None = None,
        top_k: int = 20,
    ) -> EvaluationResult:
        """
        Evaluate a retrieval strategy.

        Args:
            strategy:
                Name of the retrieval strategy.

            examples:
                Evaluation examples to evaluate.

            retriever:
                Optional retriever override.

            dataset:
                Optional EvaluationDataset. Its examples are used when
                examples is not supplied.

            top_k:
                Number of retrieval candidates requested.

        Returns:
            EvaluationResult containing aggregated metrics.

        Raises:
            ValueError:
                If evaluation data is missing or empty, top_k is invalid,
                or no retriever is available.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if examples is not None and dataset is not None:
            raise ValueError("Provide either examples or dataset, not both.")

        if dataset is not None:
            evaluation_examples = list(dataset.examples)
        elif examples is not None:
            evaluation_examples = list(examples)
        else:
            raise ValueError("Evaluation examples must be provided.")

        if not evaluation_examples:
            raise ValueError("Evaluation dataset must contain at least one example.")

        active_retriever = retriever or self.retriever

        if active_retriever is None:
            raise ValueError(
                "A retriever must be provided either to RetrievalEvaluator "
                "or to evaluate()."
            )

        # ---------------------------------------------------------
        # Retrieval result containers
        # ---------------------------------------------------------

        ranked_lists: list[list[str]] = []
        relevant_sets: list[set[str]] = []

        negative_ranked_lists: list[list[str]] = []

        # ---------------------------------------------------------
        # Run retrieval
        # ---------------------------------------------------------

        for example in evaluation_examples:
            results = active_retriever(
                example.query,
                top_k,
            )

            retrieved_ids = self._extract_chunk_ids(results)

            # Negative queries are evaluated separately.
            if example.difficulty == "negative":
                negative_ranked_lists.append(retrieved_ids)
                continue

            ranked_lists.append(retrieved_ids)

            relevant_sets.append(set(example.relevant_chunk_ids))

        # ---------------------------------------------------------
        # Standard retrieval metrics
        # ---------------------------------------------------------

        if ranked_lists:
            recall_1 = sum(
                recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    1,
                )
                for retrieved_ids, relevant_ids in zip(
                    ranked_lists,
                    relevant_sets,
                    strict=True,
                )
            ) / len(ranked_lists)

            recall_3 = sum(
                recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    3,
                )
                for retrieved_ids, relevant_ids in zip(
                    ranked_lists,
                    relevant_sets,
                    strict=True,
                )
            ) / len(ranked_lists)

            recall_5 = sum(
                recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    5,
                )
                for retrieved_ids, relevant_ids in zip(
                    ranked_lists,
                    relevant_sets,
                    strict=True,
                )
            ) / len(ranked_lists)

            recall_10 = sum(
                recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    10,
                )
                for retrieved_ids, relevant_ids in zip(
                    ranked_lists,
                    relevant_sets,
                    strict=True,
                )
            ) / len(ranked_lists)

            recall_20 = sum(
                recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    20,
                )
                for retrieved_ids, relevant_ids in zip(
                    ranked_lists,
                    relevant_sets,
                    strict=True,
                )
            ) / len(ranked_lists)

            precision_1 = sum(
                precision_at_k(
                    retrieved_ids,
                    relevant_ids,
                    1,
                )
                for retrieved_ids, relevant_ids in zip(
                    ranked_lists,
                    relevant_sets,
                    strict=True,
                )
            ) / len(ranked_lists)

            precision_3 = sum(
                precision_at_k(
                    retrieved_ids,
                    relevant_ids,
                    3,
                )
                for retrieved_ids, relevant_ids in zip(
                    ranked_lists,
                    relevant_sets,
                    strict=True,
                )
            ) / len(ranked_lists)

            precision_5 = sum(
                precision_at_k(
                    retrieved_ids,
                    relevant_ids,
                    5,
                )
                for retrieved_ids, relevant_ids in zip(
                    ranked_lists,
                    relevant_sets,
                    strict=True,
                )
            ) / len(ranked_lists)

            # Existing benchmark semantics use MRR@5.
            mrr_ranked_lists = [retrieved_ids[:5] for retrieved_ids in ranked_lists]

            mrr = mean_reciprocal_rank(
                mrr_ranked_lists,
                relevant_sets,
            )

            ndcg_5 = sum(
                ndcg_at_k(
                    retrieved_ids,
                    relevant_ids,
                    5,
                )
                for retrieved_ids, relevant_ids in zip(
                    ranked_lists,
                    relevant_sets,
                    strict=True,
                )
            ) / len(ranked_lists)

        else:
            recall_1 = 0.0
            recall_3 = 0.0
            recall_5 = 0.0
            recall_10 = 0.0
            recall_20 = 0.0

            precision_1 = 0.0
            precision_3 = 0.0
            precision_5 = 0.0

            mrr = 0.0
            ndcg_5 = 0.0

        # ---------------------------------------------------------
        # Negative-query metrics
        # ---------------------------------------------------------

        false_retrieval_rate_1 = mean_false_retrieval_rate_at_k(
            negative_ranked_lists,
            1,
        )

        false_retrieval_rate_3 = mean_false_retrieval_rate_at_k(
            negative_ranked_lists,
            3,
        )

        false_retrieval_rate_5 = mean_false_retrieval_rate_at_k(
            negative_ranked_lists,
            5,
        )

        false_retrieval_rate_10 = mean_false_retrieval_rate_at_k(
            negative_ranked_lists,
            10,
        )

        false_retrieval_rate_20 = mean_false_retrieval_rate_at_k(
            negative_ranked_lists,
            20,
        )

        # ---------------------------------------------------------
        # Build result
        # ---------------------------------------------------------

        metrics = MetricResult(
            recall_at_1=recall_1,
            recall_at_3=recall_3,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            recall_at_20=recall_20,
            precision_at_1=precision_1,
            precision_at_3=precision_3,
            precision_at_5=precision_5,
            mrr=mrr,
            ndcg_at_5=ndcg_5,
            false_retrieval_rate_at_1=(false_retrieval_rate_1),
            false_retrieval_rate_at_3=(false_retrieval_rate_3),
            false_retrieval_rate_at_5=(false_retrieval_rate_5),
            false_retrieval_rate_at_10=(false_retrieval_rate_10),
            false_retrieval_rate_at_20=(false_retrieval_rate_20),
        )

        num_examples = len(evaluation_examples)

        num_non_negative_examples = sum(
            example.difficulty != "negative" for example in evaluation_examples
        )

        num_adversarial_examples = sum(
            example.difficulty == "adversarial" for example in evaluation_examples
        )

        num_negative_examples = sum(
            example.difficulty == "negative" for example in evaluation_examples
        )

        return EvaluationResult(
            strategy=strategy,
            num_examples=num_examples,
            num_non_negative_examples=num_non_negative_examples,
            num_adversarial_examples=num_adversarial_examples,
            num_negative_examples=num_negative_examples,
            metrics=metrics,
        )
