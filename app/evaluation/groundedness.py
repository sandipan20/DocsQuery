"""
Groundedness evaluation for generated RAG answers.

The evaluator checks whether generated answer sentences are supported
by the retrieval contexts cited by the answer.

Production design:

    answer sentence
        ↓
    cited context(s)
        ↓
    focused evidence units
        ↓
    NLI entailment scoring
        ↓
    best supporting evidence
        ↓
    sentence groundedness
        ↓
    overall groundedness
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from transformers import pipeline

from app.generation.context_builder import CitationContext


@dataclass(frozen=True)
class GroundednessResult:
    """
    Result returned by groundedness evaluation.

    The object supports comparison with floats so existing tests and
    callers that previously expected a numeric score continue to work.
    """

    score: float
    grounded: bool

    def __eq__(self, other: object) -> bool:
        """Allow comparison with either another result or a numeric score."""

        if isinstance(other, GroundednessResult):
            return self.score == other.score and self.grounded == other.grounded

        if isinstance(other, (int, float)):
            return self.score == float(other)

        return NotImplemented


class GroundednessEvaluator:
    """Evaluate whether generated answer claims are supported by context."""

    # Minimum entailment probability required for a sentence to be
    # considered grounded.
    ENTAILMENT_THRESHOLD = 0.50

    # Keep evidence units small enough that the NLI model focuses on
    # the actual supporting statement rather than unrelated text.
    EVIDENCE_WINDOW_WORDS = 80

    # NLI model used for entailment / contradiction / neutral scoring.
    MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"

    def __init__(self) -> None:
        """Load the NLI model."""

        self.pipeline = pipeline(
            "text-classification",
            model=self.MODEL_NAME,
        )

    def evaluate(
        self,
        answer: str,
        contexts: list[CitationContext],
    ) -> GroundednessResult:
        """
        Evaluate whether the answer is supported by retrieved context.

        Cited sentences are evaluated only against their cited contexts.

        Uncited sentences use all available contexts as a fallback.

        Each retrieval chunk is split into focused evidence units before
        NLI evaluation so unrelated material does not confuse the model.
        """

        if not answer.strip():
            return GroundednessResult(
                score=0.0,
                grounded=False,
            )

        if not contexts:
            return GroundednessResult(
                score=0.0,
                grounded=False,
            )

        sentences = self._split_sentences(answer)

        if not sentences:
            return GroundednessResult(
                score=0.0,
                grounded=False,
            )

        context_map = {context.citation_id: context for context in contexts}

        sentence_scores: list[float] = []

        for sentence in sentences:
            citations = self._extract_citations(sentence)

            hypothesis = self._remove_citations(sentence).strip()

            if not hypothesis:
                continue

            if citations:
                candidate_contexts = [
                    context_map[citation]
                    for citation in citations
                    if citation in context_map
                ]
            else:
                candidate_contexts = contexts

            if not candidate_contexts:
                sentence_scores.append(0.0)
                continue

            best_score = self._best_entailment_score(
                hypothesis=hypothesis,
                contexts=candidate_contexts,
            )

            sentence_scores.append(best_score)

        if not sentence_scores:
            return GroundednessResult(
                score=0.0,
                grounded=False,
            )

        score = sum(sentence_scores) / len(sentence_scores)

        return GroundednessResult(
            score=score,
            grounded=score >= self.ENTAILMENT_THRESHOLD,
        )

    def _best_entailment_score(
        self,
        hypothesis: str,
        contexts: list[CitationContext],
    ) -> float:
        """
        Find the strongest entailment score across all focused evidence.

        We use the maximum entailment score instead of averaging every
        evidence unit because a retrieval chunk may contain unrelated text.
        """

        best_score = 0.0

        for context in contexts:
            evidence_units = self._split_evidence(context.result.text)

            for evidence in evidence_units:
                raw_result = self.pipeline(
                    {
                        "text": evidence,
                        "text_pair": hypothesis,
                    },
                    truncation=True,
                    max_length=512,
                )

                result = self._normalize_pipeline_result(raw_result)

                label = result["label"].lower()
                score = float(result["score"])

                if label == "entailment":
                    best_score = max(best_score, score)

        return best_score

    @staticmethod
    def _normalize_pipeline_result(
        result: object,
    ) -> dict[str, object]:
        """
        Normalize Hugging Face pipeline output.

        Depending on the pipeline invocation/model version, the result may
        be either:

            {"label": "...", "score": ...}

        or:

            [{"label": "...", "score": ...}]
        """

        if isinstance(result, list):
            if not result:
                return {
                    "label": "neutral",
                    "score": 0.0,
                }

            first = result[0]

            if not isinstance(first, dict):
                raise TypeError("Unexpected NLI pipeline result item.")

            return first

        if isinstance(result, dict):
            return result

        raise TypeError(f"Unexpected NLI pipeline result type: {type(result).__name__}")

    def _split_evidence(
        self,
        evidence: str,
    ) -> list[str]:
        """
        Split a retrieval chunk into focused evidence units.
        """

        text = evidence.strip()

        if not text:
            return []

        # Split on paragraph/newline boundaries first.
        pieces = [piece.strip() for piece in re.split(r"\n+", text) if piece.strip()]

        units: list[str] = []

        for piece in pieces:
            sentences = self._split_sentences(piece)

            if sentences:
                units.extend(sentences)
            else:
                units.append(piece)

        bounded_units: list[str] = []

        for unit in units:
            words = unit.split()

            if len(words) <= self.EVIDENCE_WINDOW_WORDS:
                bounded_units.append(unit)
                continue

            for start in range(
                0,
                len(words),
                self.EVIDENCE_WINDOW_WORDS,
            ):
                window = words[start : start + self.EVIDENCE_WINDOW_WORDS]

                bounded_units.append(" ".join(window))

        return bounded_units

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split text into sentences while keeping citations attached.

        Example:

            Python is a language. [C1] Git is version control. [C2]

        becomes:

            [
                "Python is a language. [C1]",
                "Git is version control. [C2]",
            ]
        """

        pattern = re.compile(
            r".*?[.!?](?:\s*\[C\d+\])*"
            r"(?=\s+|$)"
        )

        sentences = pattern.findall(text)

        consumed = "".join(sentences)

        remaining = text[len(consumed) :].strip()

        if remaining:
            sentences.append(remaining)

        return [sentence.strip() for sentence in sentences if sentence.strip()]

    @staticmethod
    def _extract_citations(
        sentence: str,
    ) -> list[str]:
        """Extract unique citation IDs such as C1, C2, and C3."""

        citations = re.findall(
            r"\[(C\d+)\]",
            sentence,
        )

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(citations))

    @staticmethod
    def _remove_citations(
        sentence: str,
    ) -> str:
        """Remove citation markers before sending text to NLI."""

        return re.sub(
            r"\s*\[C\d+\]",
            "",
            sentence,
        )
