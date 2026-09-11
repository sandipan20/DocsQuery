"""
Tests for the retrieval benchmark CLI.
"""

from scripts.evaluate_retrieval import METHODS, parse_args


def test_parse_args_defaults_to_all_methods(monkeypatch):
    """
    Without --method, the benchmark should evaluate all strategies.
    """

    monkeypatch.setattr("sys.argv", ["evaluate_retrieval"])

    args = parse_args()

    assert args.method is None


def test_parse_args_accepts_vector(monkeypatch):
    """
    --method vector should select the vector retriever.
    """

    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_retrieval", "--method", "vector"],
    )

    args = parse_args()

    assert args.method == "vector"


def test_parse_args_accepts_all_supported_methods(monkeypatch):
    """
    Every supported method should be accepted.
    """

    for method in METHODS:
        monkeypatch.setattr(
            "sys.argv",
            ["evaluate_retrieval", "--method", method],
        )

        args = parse_args()

        assert args.method == method
