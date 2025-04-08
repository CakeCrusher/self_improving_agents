# src/self_improving_agents/runners/data_collection_runner.py
"""Runner for collecting and processing state-action data."""
import json
from datetime import datetime
from typing import List, Optional

from ..evaluator_handler.models import EvaluatorData
from ..evaluator_handler.tracker import EvaluatorTracker
from ..models.state_action import (
    Actions,
    EvalConstant,
    EvalMetrics,
    Sample,
    StateActions,
)
from .arize_connector import ArizeConnector


class DataCollectionRunner:
    """Runner for collecting and processing state-action data."""

    def __init__(
        self,
        evaluator_tracker: EvaluatorTracker,
        arize_connector: Optional[ArizeConnector] = None,
        save_dir: Optional[str] = None,
    ):
        """Initialize the data collection runner.

        Args:
            evaluator_tracker: Tracker for evaluator data
            arize_connector: Connector for Arize telemetry
            save_dir: Directory to save state-action pairs
        """
        self.evaluator_tracker = evaluator_tracker
        self.arize_connector = arize_connector or ArizeConnector()
        self.save_dir = save_dir

    def collect_data(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = datetime.now(),
        evaluator_names: Optional[List[str]] = None,
        limit: int = 100,
    ) -> StateActions:
        """Collect data and create state-action pairs.

        Args:
            start_date: Start date for data collection
            end_date: Optional end date (defaults to now)
            evaluator_names: Names of evaluators to retrieve data for
            limit: Maximum number of telemetry records to retrieve

        Returns:
            List of state-action pairs
        """
        # Fetch telemetry data from Arize
        telemetry_df = self.arize_connector.get_telemetry_data(
            start_date=start_date, end_date=end_date, limit=limit
        )

        telemetry_json = json.loads(telemetry_df.to_json(orient="records"))

        # Get evaluator data
        # TODO: extend this to names (PLURAL)
        evaluators_data: List[EvaluatorData] = []
        if not evaluator_names:
            raise ValueError("No evaluator names provided")
        for name in evaluator_names:
            evaluator_data = self.evaluator_tracker.get(name)
            if not evaluator_data.calls:
                raise ValueError(f"No calls found for evaluator {name}")
            evaluators_data.append(evaluator_data)

        # Collect constants

        # Build actions
        system_prompt = ""
        if (
            telemetry_df.iloc[0]
            .get("attributes.llm.input_messages")[0]
            .get("message.role")
            == "system"
        ):
            system_prompt = (
                telemetry_df.iloc[0]
                .get("attributes.llm.input_messages")[0]
                .get("message.content")
            )
        else:
            # TODO: should not raise error
            raise ValueError("System prompt not found")
        model = telemetry_df.iloc[0].get("attributes.llm.model_name")
        actions = Actions(system_prompt=system_prompt, model=model)

        # Build eval_constants
        eval_constants = []
        for evaluator_data in evaluators_data:
            eval_template = evaluator_data.calls[0].template
            eval_rails = evaluator_data.calls[0].rails
            if not eval_template:
                raise ValueError(
                    f"No template found for evaluator {evaluator_data.name}"
                )
            if not eval_rails:
                raise ValueError(f"No rails found for evaluator {evaluator_data.name}")
            eval_constants.append(
                EvalConstant(
                    name=evaluator_data.name,
                    eval_template=eval_template,
                    eval_rails=eval_rails,
                )
            )

        # Collect samples
        samples = []
        for item in telemetry_json:
            chat_history = json.loads(item.get("attributes.input.value"))["messages"]
            if chat_history[0].get("role") == "system":
                chat_history = chat_history[1:]

            output_generation = item.get("attributes.llm.output_messages")[0].get(
                "message.content"
            )
            evals_metrics = []
            for evaluator_data in evaluators_data:
                eval_name = evaluator_data.name
                # # TODO: restore string templating with eval_name
                # eval_score = item.get(f'eval.{eval_name}.label')
                # eval_explanation = item.get(f'eval.{eval_name}.explanation')
                eval_score = item.get("eval.formatting_consistency_out_of_5.label")
                eval_explanation = item.get(
                    "eval.formatting_consistency_out_of_5.explanation"
                )
                evals_metrics.append(
                    EvalMetrics(
                        name=eval_name,
                        eval_score=eval_score,
                        eval_reasoning=eval_explanation,
                    )
                )

            samples.append(
                Sample(
                    chat_history=chat_history,
                    output_generation=output_generation,
                    evals=evals_metrics,
                )
            )

        state_action_pair = StateActions(
            samples=samples, actions=actions, eval_constants=eval_constants
        )

        return state_action_pair
