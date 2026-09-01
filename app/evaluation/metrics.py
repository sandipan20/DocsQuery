"""
DocsQuery - Retrieval Evaluation Metrics

Implements common information-retrieval metrics:

    Recall@K
    Precision@K
    MRR
    nDCG@K

The functions operate on chunk IDs, which means they are
independent of BM25, vector search, Qdrant, or reranking.
"""

import math


def _validate_k(k: int) -> None:
    """
    Validate a metric cutoff value.

    Args:
        k:
            Number of ranked results considered.

    Raises:
        ValueError:
            If k is not positive.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0.")


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate Recall@K.

    Formula:

        relevant retrieved in top-K
        ---------------------------
             total relevant

    Args:
        retrieved_ids:
            Ranked chunk IDs returned by the retriever.

        relevant_ids:
            Ground-truth relevant chunk IDs.

        k:
            Number of top results to evaluate.

    Returns:
        Recall value between 0.0 and 1.0.
    """

    _validate_k(k)

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    retrieved_relevant = set(top_k) & relevant_ids

    return len(retrieved_relevant) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate Precision@K.

    Formula:

        relevant retrieved in top-K
        ---------------------------
              number in top-K

    Args:
        retrieved_ids:
            Ranked chunk IDs returned by the retriever.

        relevant_ids:
            Ground-truth relevant chunk IDs.

        k:
            Number of top results to evaluate.

    Returns:
        Precision value between 0.0 and 1.0.
    """

    _validate_k(k)

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    retrieved_relevant = set(top_k) & relevant_ids

    return len(retrieved_relevant) / len(top_k)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """
    Calculate the reciprocal rank of the first relevant result.

    If the first relevant result is at rank N:

        reciprocal rank = 1 / N

    If no relevant result is found:

        reciprocal rank = 0
    """

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    ranked_lists: list[list[str]],
    relevant_sets: list[set[str]],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR is the average reciprocal rank across evaluation
    examples.

    Args:
        ranked_lists:
            One ranked retrieval list per query.

        relevant_sets:
            Ground-truth relevant IDs for each query.

    Returns:
        MRR value between 0.0 and 1.0.
    """

    if len(ranked_lists) != len(relevant_sets):
        raise ValueError("ranked_lists and relevant_sets must have the same length.")

    if not ranked_lists:
        return 0.0

    reciprocal_ranks = [
        reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )
        for retrieved_ids, relevant_ids in zip(
            ranked_lists,
            relevant_sets,
            strict=True,
        )
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def _dcg_at_k(
    relevance: list[int],
    k: int,
) -> float:
    """
    Calculate Discounted Cumulative Gain at K.

    Higher-ranked relevant results receive more weight.
    """

    _validate_k(k)

    return sum(
        relevance[position] / math.log2(position + 2)
        for position in range(min(k, len(relevance)))
    )


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate binary-relevance nDCG@K.

    Each retrieved chunk has relevance:

        1 -> relevant
        0 -> not relevant

    The score is normalized against the ideal ranking.

    Returns:
        nDCG value between 0.0 and 1.0.
    """

    _validate_k(k)

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    relevance = [1 if chunk_id in relevant_ids else 0 for chunk_id in top_k]

    actual_dcg = _dcg_at_k(
        relevance,
        k,
    )

    # Ideal ranking puts all relevant chunks first.
    ideal_relevance = [
        1
        for _ in range(
            min(
                k,
                len(relevant_ids),
            )
        )
    ]

    ideal_dcg = _dcg_at_k(
        ideal_relevance,
        k,
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg
