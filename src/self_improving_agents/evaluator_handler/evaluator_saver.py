"""Implementation of evaluator saving functionality."""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import ValidationError

from .models import EvaluatorData


class EvaluatorSaver:
    """Handles saving and retrieving evaluator configuration data."""

    def __init__(self, save_dir: Optional[str] = None):
        """Initialize the evaluator saver.

        Args:
            save_dir: Directory where evaluator data will be saved.
                     Defaults to .sia in the current working directory.
        """
        self.save_dir = Path(save_dir or os.path.join(os.getcwd(), ".sia/evaluator"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.saved_data: Dict[str, EvaluatorData] = {}

    def save_evaluator(
        self,
        evaluator_name: str,
        eval_kwargs: Optional[Dict[str, Any]] = None,
        get_telemetry_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save evaluator configuration data.

        Args:
            evaluator_name: Name of the evaluator
            eval_kwargs: Optional kwargs for evaluation
            get_telemetry_kwargs: Optional kwargs for telemetry
        """
        # Create or update evaluator data
        if evaluator_name not in self.saved_data:
            self.saved_data[evaluator_name] = EvaluatorData(name=evaluator_name)

        evaluator_data = self.saved_data[evaluator_name]

        # Update kwargs if provided
        if eval_kwargs is not None:
            evaluator_data.eval_kwargs = eval_kwargs
        if get_telemetry_kwargs is not None:
            evaluator_data.get_telemetry_kwargs = get_telemetry_kwargs

        # Save to disk
        file_path = self.save_dir / f"{evaluator_name}.json"
        with open(file_path, "w") as f:
            json.dump(evaluator_data.model_dump(), f, default=str, indent=2)

    def load_evaluator(self, evaluator_name: str) -> Optional[EvaluatorData]:
        """Load evaluator configuration data.

        Args:
            evaluator_name: Name of the evaluator

        Returns:
            EvaluatorData if found, None otherwise

        Raises:
            ValueError: If the data file exists but is invalid
        """
        # Try memory first
        if evaluator_name in self.saved_data:
            return self.saved_data[evaluator_name]

        # Try loading from disk
        file_path = self.save_dir / f"{evaluator_name}.json"
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    evaluator_data = EvaluatorData(**data)
                    self.saved_data[evaluator_name] = evaluator_data
                    return evaluator_data
            except (json.JSONDecodeError, ValidationError):
                raise ValueError(f"Failed to load evaluator data from {file_path}")

        return None
