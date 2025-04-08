from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class SnapshotData(BaseModel):
    """Model representing snapshot data stored in a file."""

    run_id: str = Field(..., description="Unique identifier for the run")
    status: Literal["running", "completed"] = Field(
        ..., description="Current status of the run"
    )
    timestamp: str = Field(
        ..., description="ISO formatted timestamp when snapshot was saved"
    )
    start_time: Optional[str] = Field(
        None, description="ISO formatted timestamp when run started"
    )
    end_time: Optional[str] = Field(
        None, description="ISO formatted timestamp when run ended"
    )
    duration_seconds: Optional[float] = Field(
        None, description="Duration of the run in seconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the run"
    )
