"""
Tests for the DocsQuery evaluation quality gate.
"""

from pathlib import Path

from app.evaluation.baseline import (
    RetrievalBaseline,
)
from app.evaluation.e2e_results import (
    EndToEndSummary,
)
from app.evaluation.quality_gate import (
    EvaluationQualityGate,
    QualityGateConfig,
)
from app.evaluation.results import (
    EvaluationResult,
    MetricResult,
)


def create_retrieval_result(
    strategy: str = "BM25",
    mrr: float = 0.8,
    recall_at_5: float = 0.8,
    ndcg_at_5: float = 0.8,
) -> EvaluationResult:
    """
    Create a predictable retrieval result.
    """

    return EvaluationResult(
        strategy=strategy,
        num_examples=10,
        metrics=MetricResult(
            recall_at_1=0.5,
            recall_at_3=0.7,
            recall_at_5=recall_at_5,
            precision_at_1=0.5,
            precision_at_3=0.4,
            precision_at_5=0.3,
            mrr=mrr,
            ndcg_at_5=ndcg_at_5,
        ),
    )


def create_summary(
    citation_validity: float = 1.0,
    citation_coverage: float = 1.0,
    groundedness: float = 0.9,
    correctness: float = 0.9,
) -> EndToEndSummary:
    """
    Create a predictable answer-evaluation summary.
    """

    return EndToEndSummary(
        dataset_version="1.1.0",
        num_examples=10,
        retrieval_success_rate=1.0,
        citation_validity_rate=citation_validity,
        average_citation_coverage=citation_coverage,
        average_groundedness=groundedness,
        grounded_answer_rate=0.9,
        average_correctness=correctness,
    )


def create_config(
    tmp_path: Path,
) -> QualityGateConfig:
    """
    Create a temporary quality-gate configuration.
    """

    path = tmp_path / "evaluation.yaml"

    path.write_text(
        """
retrieval:
  max_mrr_drop: 0.05
  max_ndcg_at_5_drop: 0.05
  max_recall_at_5_drop: 0.05

answer:
  min_citation_validity: 0.95
  min_citation_coverage: 0.90
  min_groundedness: 0.80
  min_correctness: 0.70
""",
        encoding="utf-8",
    )

    return QualityGateConfig(str(path))


def test_quality_gate_passes_good_results(
    tmp_path: Path,
):
    """
    Good retrieval and answer metrics should pass.
    """

    config = create_config(tmp_path)

    gate = EvaluationQualityGate(config)

    baseline = RetrievalBaseline(
        dataset_version="1.1.0",
        results=[create_retrieval_result()],
    )

    current = [
        create_retrieval_result(
            mrr=0.78,
            recall_at_5=0.78,
            ndcg_at_5=0.78,
        )
    ]

    result = gate.check(
        current_retrieval=current,
        baseline=baseline,
        answer_summary=create_summary(),
    )

    assert result.passed is True
    assert result.failures == []


def test_retrieval_regression_fails(
    tmp_path: Path,
):
    """
    Retrieval degradation beyond the configured tolerance
    should fail.
    """

    config = create_config(tmp_path)

    gate = EvaluationQualityGate(config)

    baseline = RetrievalBaseline(
        dataset_version="1.1.0",
        results=[create_retrieval_result(mrr=0.90)],
    )

    current = [create_retrieval_result(mrr=0.80)]

    result = gate.check(
        current_retrieval=current,
        baseline=baseline,
        answer_summary=create_summary(),
    )

    assert result.passed is False

    assert any(failure.metric == "BM25.mrr" for failure in result.failures)


def test_citation_quality_failure(
    tmp_path: Path,
):
    """
    Poor citation validity should fail.
    """

    config = create_config(tmp_path)

    gate = EvaluationQualityGate(config)

    baseline = RetrievalBaseline(
        dataset_version="1.1.0",
        results=[create_retrieval_result()],
    )

    result = gate.check(
        current_retrieval=[create_retrieval_result()],
        baseline=baseline,
        answer_summary=create_summary(
            citation_validity=0.80,
        ),
    )

    assert result.passed is False

    assert any(failure.metric == "citation_validity" for failure in result.failures)


def test_groundedness_failure(
    tmp_path: Path,
):
    """
    Poor groundedness should fail.
    """

    config = create_config(tmp_path)

    gate = EvaluationQualityGate(config)

    baseline = RetrievalBaseline(
        dataset_version="1.1.0",
        results=[create_retrieval_result()],
    )

    result = gate.check(
        current_retrieval=[create_retrieval_result()],
        baseline=baseline,
        answer_summary=create_summary(
            groundedness=0.50,
        ),
    )

    assert result.passed is False

    assert any(failure.metric == "groundedness" for failure in result.failures)
