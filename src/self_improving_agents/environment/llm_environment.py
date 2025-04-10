"""LLM environment for emulating and tracking LLM calls.

This module provides a class for emulating LLM calls in a controlled environment,
with instrumentation for tracking and analysis.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from arize.otel import register
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openinference.instrumentation.openai import OpenAIInstrumentor

from ..evaluator_handler.evaluator_saver import EvaluatorSaver
from ..models.state_action import StateActions
from ..policy import LLMPolicyUpdater
from ..runners.arize_connector import ArizeConnector
from ..runners.data_collection_runner import DataCollectionRunner
from .snapshot import EnvironmentSnapshot

logger = logging.getLogger(__name__)


class LLMEnvironment:
    """Environment for emulating and tracking LLM calls."""

    def __init__(
        self,
        project_name: str = "sia",
        arize_space_id: Optional[str] = None,
        arize_api_key: Optional[str] = None,
        arize_model_id: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        checkpoint_dir: str = ".sia/checkpoint",
    ):
        """Initialize the LLM environment.

        Args:
            project_name: Name of the project for Arize tracking
            arize_space_id: Arize space ID (defaults to ARIZE_SPACE_ID env var)
            arize_api_key: Arize API key (defaults to ARIZE_API_KEY env var)
            arize_model_id: Arize model ID (defaults to ARIZE_MODEL_ID env var)
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            checkpoint_dir: Directory for action checkpoints
        """
        # Set up environment variables
        self.project_name = project_name
        self.arize_space_id = arize_space_id or os.getenv("ARIZE_SPACE_ID")
        self.arize_api_key = arize_api_key or os.getenv("ARIZE_API_KEY")
        self.arize_model_id = (
            arize_model_id or os.getenv("ARIZE_MODEL_ID") or project_name
        )
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.checkpoint_dir = checkpoint_dir

        # Ensure we have the required environment variables
        if not self.arize_space_id:
            logger.warning("ARIZE_SPACE_ID not set, some features may not work")

        if not self.arize_api_key:
            logger.warning("ARIZE_API_KEY not set, some features may not work")

        if not self.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY not set, OpenAI client will use environment variables"
            )

        # Initialize Arize tracking
        self._initialize_arize_tracking()

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.openai_api_key)

        # Initialize other components
        evaluator_saver = EvaluatorSaver()
        arize_connector = ArizeConnector(
            space_id=self.arize_space_id,
            model_id=self.arize_model_id,
        )

        self.policy = LLMPolicyUpdater()  # for retrieving checkpoint
        self.data_collector = DataCollectionRunner(
            evaluator_saver=evaluator_saver,
            arize_connector=arize_connector,
        )  # for collecting data

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _initialize_arize_tracking(self) -> None:
        """Initialize Arize platform tracing."""
        if self.arize_space_id and self.arize_api_key:
            logger.info(
                f"Registering Arize tracer provider for project '{self.project_name}'"
            )
            # Set up OTel via Arize convenience function
            tracer_provider = register(
                space_id=self.arize_space_id,
                api_key=self.arize_api_key,
                project_name=self.project_name,
            )

            # Set up OpenAI instrumentation
            logger.info("Instrumenting OpenAI client")
            OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        else:
            logger.warning(
                "Skipping Arize tracer registration due to missing credentials"
            )

    def collect_updated_state_actions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = datetime.now(),
        evaluator_names: Optional[List[str]] = None,
        limit: int = 100,
    ) -> StateActions:
        """Collect samples from the Arize telemetry data.

        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            evaluator_names: Names of evaluators to include
            limit: Maximum number of samples to collect

        Returns:
            StateActions containing the collected samples
        """
        if not evaluator_names:
            raise ValueError("evaluator_names must be provided")

        if not start_date:
            # Default to 1 day ago
            start_date = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        if not end_date:
            end_date = datetime.now()

        logger.info(f"Collecting samples from {start_date} to {end_date}")
        state_actions = self.data_collector.collect_data(
            start_date=start_date,
            end_date=end_date,
            evaluator_names=evaluator_names,
            limit=limit,
        )

        updated_actions = self.policy.load_checkpoint()

        state_actions.actions = updated_actions

        return state_actions

    def emulate_llm_call(
        self,
        state_actions: StateActions,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emulate an LLM call using the specified actions on all provided state action samples.

        Args:
            state_actions: StateActions configuration to use
            run_id: Optional run ID for tracking

        Returns:
            Response from the LLM
        """
        # Create a snapshot
        snapshot = EnvironmentSnapshot(run_id=run_id)

        # Start tracking
        snapshot.start(
            {
                "model": state_actions.actions.model,
            }
        )
        results_metadatas = []
        model = state_actions.actions.model
        # TODO: Run this in parallel
        for sample in state_actions.samples:
            system_message = [
                {
                    "role": "system",
                    "content": state_actions.actions.system_prompt,
                }
            ]
            messages = system_message + sample.chat_history
            # print json formatted chat history
            print(json.dumps(messages, indent=4))

            params = {
                "model": model,
                "messages": messages,
            }

            try:
                # Make the actual call
                logger.info(f"Making LLM call with model {model}")
                response: ChatCompletion = self.openai_client.chat.completions.create(
                    **params
                )

                # Extract the response content
                content = response.choices[0].message.content

                # Capture result metadata
                results_metadatas.append(
                    {
                        "completion_tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "response_length": len(content) if content else 0,
                    }
                )

            except Exception as e:
                logger.error(f"Error in LLM call: {e}")
                results_metadatas.append(
                    {
                        "error": str(e),
                    }
                )

            # End tracking with result metadata
            snapshot.end(
                {
                    "results_metadatas": results_metadatas,
                }
            )

        # Return the response
        return {
            "content": content,
            "usage": response.usage.model_dump(),
            "model": model,
            "run_id": snapshot.run_id,
            "duration": snapshot.get_duration(),
        }
