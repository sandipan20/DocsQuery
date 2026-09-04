"""
DocsQuery - Evaluation Quality Gate

Compares current evaluation results against the accepted
baseline and checks absolute answer-quality requirements.

The quality gate is intentionally separate from the benchmark
runner.

Benchmark:
    "What are the current metrics?"

Quality gate:
    "Are the current metrics acceptable?"
"""

from pathlib import Path

import yaml

from app.evaluation.baseline import (
    RetrievalBaseline,
    compare_results,
)
from app.evaluation.e2e_results import (
    EndToEndSummary,
)
from app.evaluation.results import (
    EvaluationResult,
)


class QualityGateConfig:
    """
    Configuration for evaluation quality checks.
    """

    def __init__(
        self,
        file_path: str,
    ):
        """
        Load quality-gate configuration from YAML.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Quality gate configuration not found: {file_path}"
            )

        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        retrieval = data["retrieval"]
        answer = data["answer"]

        self.max_mrr_drop = float(retrieval["max_mrr_drop"])

        self.max_ndcg_at_5_drop = float(retrieval["max_ndcg_at_5_drop"])

        self.max_recall_at_5_drop = float(retrieval["max_recall_at_5_drop"])

        self.min_citation_validity = float(answer["min_citation_validity"])

        self.min_citation_coverage = float(answer["min_citation_coverage"])

        self.min_groundedness = float(answer["min_groundedness"])

        self.min_correctness = float(answer["min_correctness"])


class QualityGateFailure:
    """
    Represents one quality-gate failure.
    """

    def __init__(
        self,
        metric: str,
        message: str,
    ):
        self.metric = metric
        self.message = message


class QualityGateResult:
    """
    Final result of the quality gate.
    """

    def __init__(
        self,
        passed: bool,
        failures: list[QualityGateFailure],
    ):
        self.passed = passed
        self.failures = failures


class EvaluationQualityGate:
    """
    Applies quality requirements to evaluation results.
    """

    def __init__(
        self,
        config: QualityGateConfig,
    ):
        """
        Store quality-gate configuration.
        """

        self.config = config

    def check_retrieval(
        self,
        current: list[EvaluationResult],
        baseline: RetrievalBaseline,
    ) -> list[QualityGateFailure]:
        """
        Check retrieval metrics against the accepted baseline.

        A retrieval metric is allowed to decrease only within
        its configured tolerance.
        """

        failures = []

        comparisons = compare_results(
            current=current,
            baseline=baseline.results,
        )

        for strategy, metrics in comparisons.items():
            if metrics["mrr"] < (-self.config.max_mrr_drop):
                failures.append(
                    QualityGateFailure(
                        metric=f"{strategy}.mrr",
                        message=(f"MRR dropped by {abs(metrics['mrr']):.4f}"),
                    )
                )

            if metrics["ndcg_at_5"] < (-self.config.max_ndcg_at_5_drop):
                failures.append(
                    QualityGateFailure(
                        metric=(f"{strategy}.ndcg_at_5"),
                        message=(f"nDCG@5 dropped by {abs(metrics['ndcg_at_5']):.4f}"),
                    )
                )

            if metrics["recall_at_5"] < (-self.config.max_recall_at_5_drop):
                failures.append(
                    QualityGateFailure(
                        metric=(f"{strategy}.recall_at_5"),
                        message=(
                            f"Recall@5 dropped by {abs(metrics['recall_at_5']):.4f}"
                        ),
                    )
                )

        return failures

    def check_answer_quality(
        self,
        summary: EndToEndSummary,
    ) -> list[QualityGateFailure]:
        """
        Check absolute answer-quality requirements.
        """

        failures = []

        if summary.citation_validity_rate < self.config.min_citation_validity:
            failures.append(
                QualityGateFailure(
                    metric="citation_validity",
                    message=(
                        "Citation validity "
                        f"{summary.citation_validity_rate:.4f} "
                        "is below the required minimum "
                        f"{self.config.min_citation_validity:.4f}"
                    ),
                )
            )

        if summary.average_citation_coverage < self.config.min_citation_coverage:
            failures.append(
                QualityGateFailure(
                    metric="citation_coverage",
                    message=(
                        "Citation coverage "
                        f"{summary.average_citation_coverage:.4f} "
                        "is below the required minimum "
                        f"{self.config.min_citation_coverage:.4f}"
                    ),
                )
            )

        if summary.average_groundedness < self.config.min_groundedness:
            failures.append(
                QualityGateFailure(
                    metric="groundedness",
                    message=(
                        "Groundedness "
                        f"{summary.average_groundedness:.4f} "
                        "is below the required minimum "
                        f"{self.config.min_groundedness:.4f}"
                    ),
                )
            )

        if (
            summary.average_correctness is not None
            and summary.average_correctness < self.config.min_correctness
        ):
            failures.append(
                QualityGateFailure(
                    metric="correctness",
                    message=(
                        "Correctness "
                        f"{summary.average_correctness:.4f} "
                        "is below the required minimum "
                        f"{self.config.min_correctness:.4f}"
                    ),
                )
            )

        return failures

    def check(
        self,
        current_retrieval: list[EvaluationResult],
        baseline: RetrievalBaseline,
        answer_summary: EndToEndSummary,
    ) -> QualityGateResult:
        """
        Run all quality checks.
        """

        failures = []

        failures.extend(
            self.check_retrieval(
                current=current_retrieval,
                baseline=baseline,
            )
        )

        failures.extend(self.check_answer_quality(summary=answer_summary))

        return QualityGateResult(
            passed=not failures,
            failures=failures,
        )
