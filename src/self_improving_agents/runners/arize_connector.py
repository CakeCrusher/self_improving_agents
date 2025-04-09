# src/self_improving_agents/runners/arize_connector.py
"""Connector for retrieving data from Arize telemetry."""
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from arize.exporter import ArizeExportClient
from arize.pandas.logger import Client
from arize.utils.types import Environments


class ArizeConnector:
    """Connector for retrieving data from Arize telemetry."""

    def __init__(
        self,
        developer_key: Optional[str] = None,
        space_id: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the Arize connector.

        Args:
            developer_key: Developer key for Arize (can also be set as env var)
            space_id: Arize space ID
            model_id: Arize model ID
            api_key: Arize API key for logging evaluations
        """
        self.ARIZE_DEVELOPER_KEY = developer_key or os.getenv("ARIZE_DEVELOPER_KEY")
        self.space_id = space_id
        self.model_id = model_id
        self.client = ArizeExportClient(api_key=self.ARIZE_DEVELOPER_KEY)
        self.ARIZE_MODEL_ID = model_id or os.getenv("ARIZE_MODEL_ID")
        self.ARIZE_SPACE_ID = space_id or os.getenv("ARIZE_SPACE_ID")
        self.ARIZE_API_KEY = api_key or os.getenv("ARIZE_API_KEY")
        self.ARIZE_DEVELOPER_KEY = developer_key or os.getenv("ARIZE_DEVELOPER_KEY")

        if not self.ARIZE_DEVELOPER_KEY:
            raise ValueError("ARIZE_DEVELOPER_KEY is not set")
        if not self.ARIZE_SPACE_ID:
            raise ValueError("ARIZE_SPACE_ID is not set")
        if not self.ARIZE_MODEL_ID:
            raise ValueError("ARIZE_MODEL_ID is not set")

    def get_telemetry_data(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = datetime.now(),
        limit: int = 100,
    ) -> pd.DataFrame:
        """Retrieve telemetry data from Arize.

        Args:
            start_date: Start date for data retrieval
            end_date: Optional end date (defaults to now)
            limit: Maximum number of records to retrieve

        Returns:
            DataFrame containing telemetry data
        """
        # Exporting your dataset into a dataframe
        primary_df = self.client.export_model_to_df(
            model_id=self.ARIZE_MODEL_ID,
            space_id=self.ARIZE_SPACE_ID,
            environment=Environments.TRACING,
            start_time=start_date,
            end_time=end_date,
        )
        primary_df = primary_df.head(limit)

        return primary_df

    def upload_evaluations(
        self, primary_df: pd.DataFrame, evals_df: pd.DataFrame, eval_name: str
    ) -> None:
        """Upload evaluations to Arize.

        Args:
            evals_df: DataFrame containing evaluation results
            eval_name: Name of the evaluator
        """
        if not self.ARIZE_API_KEY:
            raise ValueError("ARIZE_API_KEY is not set")

        # ensure that the primary_df and evals_df have the same number of rows
        if primary_df.shape[0] != evals_df.shape[0]:
            raise ValueError(
                "primary_df and evals_df must have the same number of rows"
            )

        # Create a client for logging evaluations
        logging_client = Client(
            space_id=self.ARIZE_SPACE_ID,
            developer_key=self.ARIZE_DEVELOPER_KEY,
            api_key=self.ARIZE_API_KEY,
        )

        evals_df["context.span_id"] = primary_df["context.span_id"]

        # Format the column names as expected by Arize
        evals_df[f"eval.{eval_name}.label"] = evals_df["label"]
        evals_df[f"eval.{eval_name}.explanation"] = evals_df["explanation"]
        evals_df[f"eval.{eval_name}.score"] = evals_df["score"]

        # Upload the evaluations
        logging_client.log_evaluations_sync(evals_df, self.ARIZE_MODEL_ID)

        return
