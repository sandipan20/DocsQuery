"""
DocsQuery - Evaluation Result Models

Structured representations of retrieval evaluation results.
"""

from pydantic import BaseModel


class MetricResult(BaseModel):
    """
    Metrics calculated for one retrieval strategy.
    """

    recall_at_1: float

    recall_at_3: float

    recall_at_5: float

    # Deeper retrieval diagnostics.
    #
    # These values tell us whether relevant chunks exist
    # beyond the original top-5 evaluation window.
    recall_at_10: float = 0.0

    recall_at_20: float = 0.0

    precision_at_1: float

    precision_at_3: float

    precision_at_5: float

    mrr: float

    ndcg_at_5: float

    # Negative-query diagnostics.
    #
    # Lower is better. A value of 0.0 means no negative query
    # returned any result within the specified top-K window.
    false_retrieval_rate_at_1: float = 0.0

    false_retrieval_rate_at_3: float = 0.0

    false_retrieval_rate_at_5: float = 0.0

    false_retrieval_rate_at_10: float = 0.0

    false_retrieval_rate_at_20: float = 0.0


class EvaluationResult(BaseModel):
    """
    Complete result for one retrieval system.
    """

    strategy: str

    # Total number of evaluation examples.
    num_examples: int

    # All examples used for standard retrieval metrics.
    #
    # This includes normal positive and adversarial examples,
    # but excludes negative examples.
    num_non_negative_examples: int = 0

    # Number of adversarial examples.
    num_adversarial_examples: int = 0

    # Number of negative/out-of-corpus examples.
    num_negative_examples: int = 0

    metrics: MetricResult
