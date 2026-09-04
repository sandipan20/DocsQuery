"""
Unit tests for groundedness evaluation.

The real NLI model is not loaded here.
"""

from app.evaluation.groundedness import (
    GroundednessEvaluator,
)


class FakeNLIPipeline:
    """
    Fake NLI pipeline returning deterministic entailment.
    """

    def __call__(
        self,
        inputs,
        **kwargs,
    ):
        return [
            {
                "label": "entailment",
                "score": 0.99,
            }
        ]


def create_evaluator():
    """
    Create a groundedness evaluator without loading a model.
    """

    evaluator = object.__new__(GroundednessEvaluator)

    evaluator.model_name = "fake-model"
    evaluator.pipeline = FakeNLIPipeline()

    return evaluator


def test_empty_answer_is_not_grounded():
    """
    Empty answers should have groundedness 0.
    """

    evaluator = create_evaluator()

    assert (
        evaluator.evaluate(
            "",
            "Some evidence.",
        )
        == 0.0
    )


def test_empty_evidence_is_not_grounded():
    """
    Without evidence, no answer can be grounded.
    """

    evaluator = create_evaluator()

    assert (
        evaluator.evaluate(
            "Python is a language.",
            "",
        )
        == 0.0
    )


def test_entailing_answer_is_grounded():
    """
    Fake NLI entailment should produce a perfect groundedness
    score.
    """

    evaluator = create_evaluator()

    score = evaluator.evaluate(
        "Python is a language.",
        "Python is a programming language.",
    )

    assert score == 1.0
