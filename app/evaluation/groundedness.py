"""
DocsQuery - Groundedness Evaluation

Uses a Natural Language Inference (NLI) model to estimate
whether generated answer sentences are supported by retrieved
evidence.

This evaluator is intentionally part of the evaluation layer,
not the production request path.

Production RAG:
    Retrieval → Reranker → Gemini

Evaluation:
    Answer + Evidence → NLI model → Groundedness score
"""

from functools import lru_cache

from transformers import pipeline


class GroundednessEvaluator:
    """
    Evaluates whether answer statements are supported by evidence.
    """

    # Keep evidence windows reasonably small.
    #
    # The actual tokenizer limit is enforced by the NLI pipeline
    # with truncation=True and max_length=512.
    MAX_EVIDENCE_WORDS = 300

    def __init__(
        self,
        model_name: str = ("cross-encoder/nli-MiniLM2-L6-H768"),
    ):
        """
        Initialize the NLI pipeline.

        The model is loaded when this evaluator is constructed.
        """

        self.model_name = model_name

        self.pipeline = pipeline(
            "text-classification",
            model=model_name,
        )

    def evaluate(
        self,
        answer: str,
        evidence: str,
    ) -> float:
        """
        Estimate groundedness.

        Each answer sentence is compared against manageable
        evidence windows.

        A sentence is considered grounded when at least one
        evidence window is classified as entailment.

        Args:
            answer:
                Generated answer.

            evidence:
                Retrieved evidence.

        Returns:
            Fraction of answer sentences judged to be entailed
            by the evidence.
        """

        if not answer.strip():
            return 0.0

        if not evidence.strip():
            return 0.0

        sentences = self._split_sentences(answer)

        if not sentences:
            return 0.0

        evidence_windows = self._split_evidence(evidence)

        if not evidence_windows:
            return 0.0

        supported = 0

        for sentence in sentences:
            if self._sentence_is_supported(
                sentence=sentence,
                evidence_windows=evidence_windows,
            ):
                supported += 1

        return supported / len(sentences)

    def _sentence_is_supported(
        self,
        sentence: str,
        evidence_windows: list[str],
    ) -> bool:
        """
        Determine whether an answer sentence is supported by
        at least one evidence window.

        Evidence is the NLI premise and the answer sentence is
        the NLI hypothesis.
        """

        for evidence_window in evidence_windows:
            result = self.pipeline(
                {
                    "text": evidence_window,
                    "text_pair": sentence,
                },
                truncation=True,
                max_length=512,
            )

            if not result:
                continue

            # Transformers may return either a single prediction
            # dictionary or a list containing predictions.
            if isinstance(result, list):
                prediction = result[0]
            else:
                prediction = result

            label = str(prediction["label"]).lower()

            if "entail" in label:
                return True

        return False

    @classmethod
    def _split_evidence(
        cls,
        evidence: str,
    ) -> list[str]:
        """
        Split large evidence into manageable word-based windows.

        Word-based windows keep the amount of evidence bounded,
        while the tokenizer-level truncation in the NLI call
        provides the final protection against model limits.
        """

        words = evidence.split()

        if not words:
            return []

        return [
            " ".join(words[start : start + cls.MAX_EVIDENCE_WORDS])
            for start in range(
                0,
                len(words),
                cls.MAX_EVIDENCE_WORDS,
            )
        ]

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:
        """
        Perform simple sentence splitting.
        """

        import re

        return [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                text.strip(),
            )
            if sentence.strip()
        ]


@lru_cache
def get_groundedness_evaluator() -> GroundednessEvaluator:
    """
    Return a cached evaluator.

    This prevents multiple NLI model instances from being
    created within the same evaluation process.
    """

    return GroundednessEvaluator()
