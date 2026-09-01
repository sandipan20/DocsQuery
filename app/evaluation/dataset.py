"""
DocsQuery - Evaluation Dataset Loader

Loads and validates the retrieval evaluation dataset.

Flow:

    retrieval_dataset.json
            ↓
        JSON parsing
            ↓
      Pydantic validation
            ↓
      EvaluationDataset
"""

import json
from pathlib import Path

from app.evaluation.models import EvaluationDataset


def load_evaluation_dataset(
    file_path: str,
) -> EvaluationDataset:
    """
    Load and validate an evaluation dataset.

    Args:
        file_path:
            Path to the evaluation JSON file.

    Returns:
        A validated EvaluationDataset object.

    Raises:
        FileNotFoundError:
            If the dataset file does not exist.

        json.JSONDecodeError:
            If the file contains invalid JSON.

        pydantic.ValidationError:
            If the JSON structure is invalid.
    """

    # Convert the string path into a Path object.
    path = Path(file_path)

    # Fail early with a clear error if the file is missing.
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {file_path}")

    # Read the JSON file.
    raw_text = path.read_text(encoding="utf-8")

    # Parse the JSON into Python objects.
    data = json.loads(raw_text)

    # Validate the parsed data using our Pydantic model.
    return EvaluationDataset.model_validate(data)
