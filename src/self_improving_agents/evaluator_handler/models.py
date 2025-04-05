"""Pydantic models for evaluator handler."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvaluatorCall(BaseModel):
    """Model representing a single call to an evaluator function."""

    timestamp: datetime = Field(default_factory=datetime.now)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    template: Optional[str] = None
    rails: Optional[List[Any]] = None


class EvaluatorData(BaseModel):
    """Model representing all data for a specific evaluator."""

    name: str
    calls: List[EvaluatorCall] = Field(default_factory=list)
