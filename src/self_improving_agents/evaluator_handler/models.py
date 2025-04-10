"""Pydantic models for evaluator handler."""
from typing import Any, Dict

from pydantic import BaseModel, Field


class EvaluatorData(BaseModel):
    """Model representing all data for a specific evaluator."""

    name: str
    eval_kwargs: Dict[str, Any] = Field(default_factory=dict)
    get_telemetry_kwargs: Dict[str, Any] = Field(default_factory=dict)
