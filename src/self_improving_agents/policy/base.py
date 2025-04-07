"""Base optimizer classes for prompt optimization.

This module defines the base optimizer interfaces that concrete optimizers
will implement.
"""
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from ..models.state_action import Actions, StateActions


class BasePolicy(ABC):
    """Abstract base class for policy update strategies."""

    @abstractmethod
    def update(self, state_actions: StateActions) -> Actions:
        """Update the policy based on collected state-action data.

        Args:
            state_actions: The state-action pairs used for policy update

        Returns:
            Updated actions (system prompt, model parameters, etc.)
        """
        pass

    def save_checkpoint(self, actions: Actions) -> str:
        """Save actions as a JSON checkpoint.

        Args:
            actions: The actions to save

        Returns:
            The path to the saved checkpoint
        """
        # Create directory if it doesn't exist
        date_str = datetime.now().strftime("%Y%m%d%H%M%S")
        print("SAVING CHECKPOINT")

        # Create the full path correctly
        checkpoint_dir = ".sia/checkpoint"
        checkpoint_file = f"actions_{date_str}.json"
        checkpoint_path = f"{checkpoint_dir}/{checkpoint_file}"

        # Create directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save as JSON
        with open(checkpoint_path, "w") as f:
            json.dump(actions.model_dump(), f, indent=2)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Actions:
        """Load actions from a JSON checkpoint.
        If no checkpoint path is provided, the latest checkpoint will be loaded.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if checkpoint_path is None:
            checkpoint_dir = ".sia/checkpoint"
            checkpoint_files = [
                f for f in os.listdir(checkpoint_dir) if f.startswith("actions_")
            ]
            if not checkpoint_files:
                raise FileNotFoundError("No checkpoint files found")
            checkpoint_path = os.path.join(checkpoint_dir, sorted(checkpoint_files)[-1])

        with open(checkpoint_path, "r") as f:
            return Actions(**json.load(f))
