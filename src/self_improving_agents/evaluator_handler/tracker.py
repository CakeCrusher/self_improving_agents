"""Implementation of evaluator tracking functionality."""
import functools
import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from pydantic import ValidationError

from .models import EvaluatorCall, EvaluatorData


class EvaluatorTracker:
    """Handles tracking, storing, and retrieving evaluator data."""

    def __init__(self, save_dir: Optional[str] = None):
        """Initialize the evaluator tracker.

        Args:
            save_dir: Directory where tracking data will be saved.
                     Defaults to .sia in the current working directory.
        """
        self.save_dir = Path(save_dir or os.path.join(os.getcwd(), ".sia"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tracked_data: Dict[str, EvaluatorData] = {}

    def track(self, evaluator: Callable, name: Optional[str] = None) -> Callable:
        """Wrap an evaluator function for tracking.

        Args:
            evaluator: The evaluator function to track
            name: Optional name for the evaluator

        Returns:
            A wrapped version of the evaluator function
        """
        name = name or getattr(evaluator, "__name__", "unknown_evaluator")

        # Initialize data store for this evaluator if it doesn't exist
        if name and name not in self.tracked_data:
            self.tracked_data[name] = EvaluatorData(name=name)

        # Create a wrapper function that calls and tracks the evaluator
        @functools.wraps(evaluator)
        def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
            # Get call timestamp and bound args
            call_time = datetime.now()
            bound_args = self._get_bound_args(evaluator, *args, **kwargs)
            print("BOUND ARGS: ", bound_args)

            # Extract template and rails
            template = str(kwargs.get("template") or bound_args.get("template"))
            rails = list(kwargs.get("rails") or bound_args.get("rails") or [])
            if not template:
                raise ValueError("Template is required")
            if not rails:
                raise ValueError("Rails are required")

            # Call evaluator and get output
            output = evaluator(*args, **kwargs)

            # Extract metadata from output dataframe if present
            metadata = {}
            if isinstance(output, pd.DataFrame) and not output.empty:
                metadata = output.iloc[0].to_dict()

            # Create and track call data
            call_data = EvaluatorCall(
                timestamp=call_time,
                inputs=bound_args,
                output=output,
                metadata=metadata,
                template=template,
                rails=rails,
            )
            if not name:
                raise ValueError("Evaluator name is required")
            self.add_call(name, call_data)

            return output

        # # Add tracking marker attributes
        # wrapper.is_tracked = True
        # wrapper.original_evaluator = evaluator
        # wrapper.evaluator_name = name

        return wrapper

    def _get_bound_args(
        self, evaluator: Callable, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert args and kwargs to a dictionary of parameter names and values."""
        try:
            # Get the signature of the evaluator function
            sig = inspect.signature(evaluator)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Convert to a dict, handling non-serializable objects
            result = {}
            for param_name, param_value in bound.arguments.items():
                # Convert pandas DataFrames to dict records for serialization
                if isinstance(param_value, pd.DataFrame):
                    result[param_name] = {
                        "dataframe_records": param_value.to_dict(orient="records")
                    }
                else:
                    try:
                        # Check if value is json serializable
                        json.dumps(param_value)
                        result[param_name] = param_value
                    except (TypeError, OverflowError):
                        # If not serializable, store string representation
                        result[param_name] = {"type": type(param_value).__name__}

            return result
        except (ValueError, TypeError):
            # If we can't inspect the signature, just record position-based args
            return {
                "args": [
                    str(arg)
                    if not isinstance(arg, (int, float, str, bool, list, dict))
                    else arg
                    for arg in args
                ],
                "kwargs": kwargs,
            }

    def add_call(self, evaluator_name: str, call_data: EvaluatorCall) -> None:
        """Add a call record to the tracked data.

        Args:
            evaluator_name: Name of the evaluator
            call_data: Data from the evaluator call
        """
        if evaluator_name not in self.tracked_data:
            self.tracked_data[evaluator_name] = EvaluatorData(name=evaluator_name)

        self.tracked_data[evaluator_name].calls.append(call_data)

        # Auto-save after each call
        self.save(evaluator_name)

    def save(self, evaluator_name: Optional[str] = None) -> None:
        """Save tracked evaluator data to disk.

        Args:
            evaluator_name: Optional name of specific evaluator to save.
                           If None, saves all evaluators.
        """
        if evaluator_name:
            self._save_evaluator(evaluator_name)
        else:
            for name in self.tracked_data:
                self._save_evaluator(name)

    def _save_evaluator(self, evaluator_name: str) -> None:
        """Save data for a specific evaluator.

        Args:
            evaluator_name: Name of the evaluator to save
        """
        if evaluator_name not in self.tracked_data:
            return

        data = self.tracked_data[evaluator_name]
        file_path = self.save_dir / f"evaluator-{evaluator_name}.json"

        with open(file_path, "w") as f:
            json.dump(data.model_dump(), f, default=str, indent=2)

    def get(self, evaluator_name: str) -> Optional[EvaluatorData]:
        """Get tracked data for a specific evaluator.

        Args:
            evaluator_name: Name of the evaluator

        Returns:
            List of call records for the evaluator
        """
        # Try to load from memory first
        if evaluator_name in self.tracked_data:
            evaluator_data = self.tracked_data[evaluator_name]

        # If not in memory, try to load from disk
        file_path = self.save_dir / f"evaluator-{evaluator_name}.json"
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    evaluator_data = EvaluatorData(**data)
                    self.tracked_data[evaluator_name] = evaluator_data
                    return evaluator_data
            except (json.JSONDecodeError, ValidationError):
                raise ValueError(f"Failed to load evaluator data from {file_path}")

        return None


def is_tracked_evaluator(evaluator: Callable) -> bool:
    """Check if an evaluator is tracked.

    Args:
        evaluator: The evaluator function to check

    Returns:
        True if the evaluator is tracked, False otherwise
    """
    return getattr(evaluator, "is_tracked", False)
