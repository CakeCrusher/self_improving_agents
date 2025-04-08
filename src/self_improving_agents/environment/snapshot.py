"""Environment snapshot for tracking LLM call runs.

This module provides a class for creating and tracking environment snapshots
during LLM call runs, allowing for analysis of start and end times.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from ..models.snapshot import SnapshotData


class EnvironmentSnapshot:
    """Snapshot of the environment state during LLM call runs."""

    def __init__(
        self, run_id: Optional[str] = None, snapshot_dir: str = ".sia/snapshots"
    ):
        """Initialize an environment snapshot.

        Args:
            run_id: Unique identifier for the run (defaults to timestamp)
            snapshot_dir: Directory to save snapshots
        """
        self.run_id = run_id or datetime.now().strftime("%Y%m%d%H%M%S")
        self.snapshot_dir = snapshot_dir
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}

        # Create the snapshots directory if it doesn't exist
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def start(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start the snapshot tracking.

        Args:
            metadata: Optional metadata about the run
        """
        self.start_time = datetime.now()
        if metadata:
            self.metadata.update(metadata)

        # Save initial snapshot
        self._save_snapshot()

    def end(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """End the snapshot tracking.

        Args:
            metadata: Optional metadata about the run results
        """
        self.end_time = datetime.now()
        if metadata:
            self.metadata.update(metadata)

        # Save final snapshot
        self._save_snapshot()

    def _save_snapshot(self) -> str:
        """Save the current snapshot to a file.

        Returns:
            Path to the saved snapshot file
        """
        # Get the current status of the run
        status: Literal["running", "completed"] = (
            "completed" if self.end_time else "running"
        )

        # Create the snapshot data using the Pydantic model
        snapshot_data = SnapshotData(
            run_id=self.run_id,
            status=status,
            timestamp=datetime.now().isoformat(),
            start_time=self.start_time.isoformat() if self.start_time else None,
            end_time=self.end_time.isoformat() if self.end_time else None,
            duration_seconds=(
                (self.end_time - self.start_time).total_seconds()
                if self.end_time and self.start_time
                else None
            ),
            metadata=self.metadata,
        )

        # Create a single filename for the run
        filename = f"{self.run_id}.json"
        filepath = os.path.join(self.snapshot_dir, filename)

        # Save as JSON
        with open(filepath, "w") as f:
            f.write(snapshot_data.model_dump_json(indent=2))

        return filepath

    def get_duration(self) -> Optional[float]:
        """Get the duration of the run in seconds.

        Returns:
            Duration in seconds or None if run not completed
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def load(
        self, filepath: Optional[str] = None, snapshot_dir: str = ".sia/snapshots"
    ) -> SnapshotData:
        """Load a snapshot from a file.

        If no filepath is provided, the most recent snapshot file in the
        snapshot_dir will be loaded.

        Args:
            filepath: Optional path to snapshot file
            snapshot_dir: Directory containing snapshot files (used if filepath is None)

        Returns:
            SnapshotData object containing the loaded snapshot data

        Raises:
            FileNotFoundError: If no snapshot files exist or the specified file doesn't exist
        """
        # If no filepath provided, find the most recent snapshot
        if filepath is None:
            if not os.path.exists(snapshot_dir):
                raise FileNotFoundError(
                    f"Snapshot directory {snapshot_dir} does not exist"
                )

            # Get all json files in the snapshot directory
            snapshot_files = [
                os.path.join(snapshot_dir, f)
                for f in os.listdir(snapshot_dir)
                if f.endswith(".json")
            ]

            if not snapshot_files:
                raise FileNotFoundError(f"No snapshot files found in {snapshot_dir}")

            # Sort by modification time (most recent last)
            snapshot_files.sort(key=lambda f: os.path.getmtime(f))
            filepath = snapshot_files[-1]

        print(f"Loading snapshot from {filepath}")

        # Ensure the file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Snapshot file {filepath} does not exist")

        # Load the snapshot data and parse it with the Pydantic model
        with open(filepath, "r") as f:
            data = json.load(f)
            return SnapshotData(**data)
