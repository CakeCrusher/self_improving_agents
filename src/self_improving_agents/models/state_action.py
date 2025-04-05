# src/self_improving_agents/models/state_action.py
"""State-action pair models for policy learning."""
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class EvalMetrics(BaseModel):
    """Evaluation metrics from a single evaluation."""

    name: str
    eval_score: Any
    eval_reasoning: str


class Sample(BaseModel):
    """A sample interaction with associated evaluations."""

    chat_history: List[Dict[str, Any]]
    output_generation: str
    evals: List[EvalMetrics]


class Actions(BaseModel):
    """An action that can be taken by the system."""

    system_prompt: str
    model: str


class EvalConstant(BaseModel):
    """Constants used for evaluation."""

    name: str
    eval_template: str
    eval_rails: List[Any]


class StateActions(BaseModel):
    """A state-action pair for policy learning."""

    id: str = Field(default_factory=lambda: datetime.now().isoformat())
    timestamp: datetime = Field(default_factory=datetime.now)
    samples: List[Sample]
    actions: Actions
    eval_constants: List[EvalConstant]
