"""
DocsQuery - Retrieval Diagnostics

Contains structured information about individual evaluation
examples.

Unlike aggregate metrics, diagnostics tell us exactly which
queries succeeded or failed for each retrieval strategy.
"""

from pydantic import BaseModel


class StrategyDiagnostic(BaseModel):
    """
    Retrieval outcome for one strategy.
    """

    strategy: str

    # Ranked chunk IDs returned by the strategy.
    retrieved_chunk_ids: list[str]

    # Whether at least one ground-truth chunk appeared
    # in the returned ranking.
    hit: bool


class QueryDiagnostic(BaseModel):
    """
    Detailed evaluation information for one query.
    """

    example_id: str

    query: str

    # Ground-truth relevant chunks.
    relevant_chunk_ids: list[str]

    # Result for each retrieval strategy.
    strategies: list[StrategyDiagnostic]


def analyze_example(
    example_id: str,
    query: str,
    relevant_chunk_ids: list[str],
    strategy_results: dict[str, list[str]],
) -> QueryDiagnostic:
    """
    Build diagnostics for a single evaluation example.

    Args:
        example_id:
            Unique evaluation example ID.

        query:
            Evaluation query.

        relevant_chunk_ids:
            Ground-truth relevant chunks.

        strategy_results:
            Mapping of strategy name to ranked chunk IDs.

    Returns:
        Structured diagnostic information.
    """

    relevant = set(relevant_chunk_ids)

    diagnostics = []

    for strategy, retrieved_ids in strategy_results.items():
        # A hit occurs when at least one returned chunk
        # belongs to the ground-truth relevant set.
        hit = bool(set(retrieved_ids) & relevant)

        diagnostics.append(
            StrategyDiagnostic(
                strategy=strategy,
                retrieved_chunk_ids=retrieved_ids,
                hit=hit,
            )
        )

    return QueryDiagnostic(
        example_id=example_id,
        query=query,
        relevant_chunk_ids=relevant_chunk_ids,
        strategies=diagnostics,
    )
