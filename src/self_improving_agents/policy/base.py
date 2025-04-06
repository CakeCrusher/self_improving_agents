"""Base optimizer classes for prompt optimization.

This module defines the base optimizer interfaces that concrete optimizers
will implement.
"""
from abc import ABC, abstractmethod

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
