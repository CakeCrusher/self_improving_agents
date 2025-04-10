# src/self_improving_agents/runners/data_collection_runner.py
"""Runner for collecting and processing state-action data."""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from self_improving_agents.evaluator_handler.evaluator_saver import EvaluatorSaver

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
        evaluator_saver: EvaluatorSaver,
        arize_connector: Optional[ArizeConnector] = None,
        save_dir: Optional[str] = None,
    ):
        """Initialize the data collection runner.

        Args:
            evaluator_saver: Saver for evaluator data
            arize_connector: Connector for Arize telemetry
            save_dir: Directory to save state-action pairs
        """
        self.evaluator_saver = evaluator_saver
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

        if not telemetry_json:
            raise ValueError("No telemetry data found")

        # Get evaluator data
        # TODO: extend this to names (PLURAL)
        evaluator_constants: List[EvalConstant] = self.discover_evaluators(
            evaluator_names=evaluator_names,
            telemetry_json=telemetry_json,
        )

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
            for evaluator_data in evaluator_constants:
                eval_name = evaluator_data.name
                # # TODO: restore string templating with eval_name
                # eval_score = item.get(f'eval.{eval_name}.label')
                # eval_explanation = item.get(f'eval.{eval_name}.explanation')
                eval_score = item.get(f"eval.{eval_name}.score")
                eval_explanation = item.get(f"eval.{eval_name}.explanation")
                eval_label = item.get(f"eval.{eval_name}.label")
                evals_metrics.append(
                    EvalMetrics(
                        name=eval_name,
                        eval_score=eval_score,
                        eval_reasoning=eval_explanation,
                        eval_label=eval_label,
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
            samples=samples, actions=actions, eval_constants=evaluator_constants
        )

        return state_action_pair

    def discover_evaluators(
        self,
        evaluator_names: Optional[List[str]] = None,
        telemetry_json: Optional[List[Dict[str, Any]]] = None,
    ) -> List[EvalConstant]:
        """Discover evaluator data from telemetry data.

        Args:
            evaluator_names: Optional list of evaluator names to search for in telemetry data
            telemetry_json: Optional telemetry data to search through

        Returns:
            List of EvalConstant objects containing discovered evaluator data
        """
        evaluators_data: List[EvalConstant] = []

        if not telemetry_json:
            raise ValueError("No telemetry data found")

        if evaluator_names:
            for name in evaluator_names:
                try:
                    # Step 1: Search through evaluator data if names provided
                    evaluator_data = self.evaluator_saver.load_evaluator(name)
                    if evaluator_data:
                        evaluators_data.append(
                            EvalConstant(
                                name=name,
                                eval_template=evaluator_data.eval_kwargs.get(
                                    "template"
                                ),
                                eval_rails=evaluator_data.eval_kwargs.get("rails"),
                            )
                        )
                        continue

                    # Step 2: Search through telemetry data if available
                    for key, val in telemetry_json[0].items():
                        if key.startswith(f"eval.{name}."):
                            evaluators_data.append(EvalConstant(name=name))
                            break

                except ValueError:
                    # Continue if evaluator not found in tracker
                    continue

        return evaluators_data
