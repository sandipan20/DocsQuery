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

    precision_at_1: float

    precision_at_3: float

    precision_at_5: float

    mrr: float

    ndcg_at_5: float


class EvaluationResult(BaseModel):
    """
    Complete result for one retrieval system.
    """

    strategy: str

    num_examples: int

    metrics: MetricResult
