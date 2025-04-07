"""
Ephemeral validator module for quick validation of policy checkpoints.

This module provides functionality to:
1. Collect state-action data using DataCollectionRunner
2. Load the latest policy checkpoint
3. Generate samples using the loaded policy
4. Run evaluations on those samples without tracking
5. Calculate the average evaluation score
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from phoenix.evals import OpenAIModel, llm_classify

from ..evaluator_handler.tracker import EvaluatorTracker
from ..models.state_action import Actions, StateActions
from ..policy.base import BasePolicy
from ..runners.arize_connector import ArizeConnector
from ..runners.data_collection_runner import DataCollectionRunner


class EphemeralValidator:
    """Performs ephemeral validation of policy checkpoints."""

    def __init__(
        self,
        evaluator_tracker: Optional[EvaluatorTracker] = None,
        arize_connector: Optional[ArizeConnector] = None,
        base_policy: Optional[BasePolicy] = None,
        openai_api_key: Optional[str] = None,
    ):
        """Initialize the ephemeral validator.

        Args:
            evaluator_tracker: Tracker for evaluator data
            arize_connector: Connector for Arize telemetry
            base_policy: Base policy for loading checkpoints
            openai_api_key: OpenAI API key for generating completions
        """
        self.evaluator_tracker = evaluator_tracker or EvaluatorTracker()
        self.arize_connector = arize_connector or ArizeConnector()
        self.data_collector = DataCollectionRunner(
            evaluator_tracker=self.evaluator_tracker,
            arize_connector=self.arize_connector,
        )
        self.base_policy = base_policy or BasePolicy()
        self.openai_client = OpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )

    def collect_state_actions(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        evaluator_names: Optional[List[str]] = None,
        limit: int = 100,
    ) -> StateActions:
        """Collect state-action data using DataCollectionRunner.

        Args:
            start_date: Start date for data collection
            end_date: Optional end date (defaults to now)
            evaluator_names: Names of evaluators to retrieve data for
            limit: Maximum number of telemetry records to retrieve

        Returns:
            StateActions object containing collected data
        """
        return self.data_collector.collect_data(
            start_date=start_date,
            end_date=end_date or datetime.now(),
            evaluator_names=evaluator_names,
            limit=limit,
        )

    def load_latest_policy(self) -> Actions:
        """Load the latest policy checkpoint.

        Returns:
            Actions object containing the latest policy
        """
        return self.base_policy.load_checkpoint()

    def generate_samples(
        self,
        actions: Actions,
        systemless_chat_histories: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, str]]:
        """Generate samples using the loaded policy.

        Args:
            actions: Actions object containing policy parameters
            inputs: List of input prompts
            limit: Maximum number of samples to generate

        Returns:
            List of dictionaries containing original input and generated output
        """
        samples = []
        for i, chat_history in enumerate(systemless_chat_histories):
            response = self.openai_client.chat.completions.create(
                model=actions.model,
                messages=[
                    {"role": "system", "content": actions.system_prompt},
                    *chat_history,
                ],
            )
            output_text = response.choices[0].message.content
            samples.append({"ORIGINAL_TEXT": input_text, "REWRITTEN_TEXT": output_text})
            print(f"Generated sample {i+1} of {limit}")

        return samples

    def evaluate_samples(
        self,
        samples: List[Dict[str, str]],
        eval_template: str,
        eval_rails: List[str],
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> pd.DataFrame:
        """Run evaluations on samples without tracking.

        Args:
            samples: List of dictionaries containing original and rewritten text
            eval_template: Evaluation template for LLM classification
            eval_rails: Evaluation rails for LLM classification
            model_name: Model name for evaluation
            temperature: Temperature for evaluation model

        Returns:
            DataFrame containing evaluation results
        """
        # Create dataframe from samples
        df = pd.DataFrame(samples)

        # Create model for evaluation
        model = OpenAIModel(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Run evaluations
        evals_df = llm_classify(
            dataframe=df,
            model=model,
            template=eval_template,
            rails=eval_rails,
            provide_explanation=True,
        )

        return evals_df

    def calculate_average_score(self, evals_df: pd.DataFrame) -> float:
        """Calculate the average evaluation score.

        Args:
            evals_df: DataFrame containing evaluation results

        Returns:
            Average evaluation score
        """
        return evals_df["label"].astype(float).mean()

    def run_validation(
        self,
        start_date: datetime,
        inputs: List[str],
        evaluator_name: str = "formatting_classify",
        limit: int = 10,
        model_name: str = "gpt-4o-mini",
    ) -> Tuple[float, pd.DataFrame]:
        """Run the complete validation process.

        Args:
            start_date: Start date for data collection
            inputs: List of input prompts
            evaluator_name: Name of evaluator to use
            limit: Maximum number of samples to generate
            model_name: Model name for evaluation

        Returns:
            Tuple containing average score and evaluation DataFrame
        """
        # Collect state-action data
        state_actions = self.collect_state_actions(
            start_date=start_date,
            evaluator_names=[evaluator_name],
            limit=3,
        )

        # Load latest policy checkpoint
        actions = self.load_latest_policy()

        # Get evaluation template and rails
        eval_constant = next(
            (ec for ec in state_actions.eval_constants if ec.name == evaluator_name),
            None,
        )
        if not eval_constant:
            raise ValueError(f"Evaluator {evaluator_name} not found in state actions")

        # Generate samples
        samples = self.generate_samples(
            actions=actions,
            systemless_chat_histories=[
                sample.chat_history[1:] for sample in state_actions.samples
            ],
        )

        # Run evaluations
        evals_df = self.evaluate_samples(
            samples=samples,
            eval_template=eval_constant.eval_template,
            eval_rails=eval_constant.eval_rails,
            model_name=model_name,
        )

        # Calculate average score
        avg_score = self.calculate_average_score(evals_df)

        return avg_score, evals_df
