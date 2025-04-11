"""
Eval pipeline for formulaic evals
"""
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

import pandas as pd
from arize.pandas.logger import Client

from ..runners.arize_connector import ArizeConnector
from .evaluator_saver import EvaluatorSaver


class EvalPipeline:
    """Pipeline for running evaluations with tracking."""

    def __init__(
        self,
        evaluator_saver: EvaluatorSaver,
        arize_connector: ArizeConnector,
    ):
        """Initialize the eval pipeline.

        Args:
            evaluator_saver: Saver for evaluator functions
            arize_connector: Optional connector for Arize integration
        """
        self.evaluator_saver = evaluator_saver
        self.arize_connector = arize_connector

    def run_pipeline(
        self,
        evaluator: Callable,
        evaluator_name: str,
        evaluator_kwargs: Dict[str, Any],
        get_telemetry_kwargs: Dict[str, Any],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        upsert: bool = False,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Execute complete evaluation pipeline.

        Args:
            evaluator: The evaluation function to run
            evaluator_name: Name of the evaluator
            evaluator_kwargs: Arguments for the evaluator
            get_telemetry_kwargs: Arguments for getting telemetry data
            upsert: Whether to upload results to Arize
            limit: Number of samples to run the evaluation on

        Returns:
            DataFrame containing evaluation results
        """
        if start_date is not None:
            get_telemetry_kwargs["start_time"] = start_date
        else:
            get_telemetry_kwargs["start_time"] = datetime.now() - timedelta(days=7)
        if end_date is not None:
            get_telemetry_kwargs["end_time"] = end_date
        else:
            get_telemetry_kwargs["end_time"] = datetime.now()

        # Get primary data from Arize
        primary_df = self.arize_connector.client.export_model_to_df(
            **get_telemetry_kwargs
        )

        # limit the number of samples to run the evaluation on
        primary_df = primary_df.tail(limit)

        # Run evaluation
        evals_df = evaluator(**evaluator_kwargs, dataframe=primary_df)

        # Add OpenInference attributes
        evals_df["context.span_id"] = primary_df["context.span_id"]
        if "score" in evals_df.columns:
            evals_df[f"eval.{evaluator_name}.score"] = evals_df["score"]
        if "label" in evals_df.columns:
            evals_df[f"eval.{evaluator_name}.label"] = evals_df["label"]
        if "explanation" in evals_df.columns:
            evals_df[f"eval.{evaluator_name}.explanation"] = evals_df["explanation"]

        # Save evaluator run
        self.evaluator_saver.save_evaluator(
            evaluator_name=evaluator_name,
            eval_kwargs=evaluator_kwargs,
            get_telemetry_kwargs=get_telemetry_kwargs,
        )

        # Upload to Arize if requested
        if upsert:
            logger_client = Client(
                space_id=self.arize_connector.ARIZE_SPACE_ID,
                developer_key=self.arize_connector.ARIZE_DEVELOPER_KEY,
                api_key=self.arize_connector.ARIZE_API_KEY,
            )
            logger_client.log_evaluations_sync(
                dataframe=evals_df, model_id=get_telemetry_kwargs["model_id"]
            )

        return evals_df
