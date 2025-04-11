"""Orchestrator for streamlining the self-improvement workflow.

This module provides an orchestrator that ties together the evaluation pipeline,
environment emulation, and policy updating components to streamline the
self-improvement workflow.
"""
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from phoenix.evals import OpenAIModel

from ..environment.llm_environment import LLMEnvironment
from ..models.state_action import Actions, StateActions
from ..policy.llm_policy_updater import LLMPolicyUpdater

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Orchestrator that streamlines the self-improvement workflow."""

    def __init__(
        self,
        arize_space_id: Optional[str] = None,
        arize_api_key: Optional[str] = None,
        arize_model_id: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        checkpoint_dir: str = ".sia/checkpoint",
    ):
        """Initialize the workflow orchestrator.

        Args:
            arize_space_id: Arize space ID (defaults to ARIZE_SPACE_ID env var)
            arize_api_key: Arize API key (defaults to ARIZE_API_KEY env var)
            arize_model_id: Arize model ID (defaults to ARIZE_MODEL_ID env var)
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            checkpoint_dir: Directory for action checkpoints
        """
        # Initialize the environment which includes most of the components we need
        self.environment = LLMEnvironment(
            arize_space_id=arize_space_id,
            arize_api_key=arize_api_key,
            arize_model_id=arize_model_id,
            openai_api_key=openai_api_key,
            checkpoint_dir=checkpoint_dir,
        )

        # For easy access to components
        self.policy_updater = LLMPolicyUpdater()

        logger.info("Workflow orchestrator initialized")

    # def run_eval_pipeline(
    #     self,
    #     evaluator: Callable,
    #     evaluator_name: str,
    #     evaluator_kwargs: Dict[str, Any],
    #     get_telemetry_kwargs: Dict[str, Any],
    #     upsert: bool = False,
    # ) -> None:
    #     """Run the evaluation pipeline to store eval information.

    #     Args:
    #         evaluator: The evaluation function to run
    #         evaluator_name: Name of the evaluator
    #         evaluator_kwargs: Arguments for the evaluator
    #         get_telemetry_kwargs: Arguments for getting telemetry data
    #         upsert: Whether to upload results to Arize
    #     """
    #     logger.info(f"Running evaluation pipeline for evaluator: {evaluator_name}")

    #     logger.info(f"Evaluation pipeline completed for: {evaluator_name}")

    # def validate_baseline(
    #     self,
    #     start_date: datetime,
    #     end_date: datetime,
    #     evaluator_names: List[str],
    #     limit: int = 100,
    #     run_id: Optional[str] = None,
    # ) -> StateActions:
    #     """Validate the baseline performance within a time window.

    #     Args:
    #         start_date: Start date for data collection
    #         end_date: End date for data collection
    #         evaluator_names: Names of evaluators to include
    #         limit: Maximum number of samples to collect
    #         run_id: Optional run ID for tracking

    #     Returns:
    #         StateActions containing the baseline samples and metrics
    #     """
    #     logger.info(f"Validating baseline from {start_date} to {end_date}")

    #     # Collect state actions from the specified time window
    #     state_actions = self.environment.data_collector.collect_data(
    #         start_date=start_date,
    #         end_date=end_date,
    #         evaluator_names=evaluator_names,
    #         limit=limit,
    #     )

    #     # Run emulation with the original actions
    #     logger.info(f"Running baseline emulation for {len(state_actions.samples)} samples")
    #     self.environment.emulate_llm_call(state_actions, run_id=run_id)

    #     return state_actions

    def update_policy(
        self,
        state_actions: StateActions,
        checkpoint: bool = True,
    ) -> Actions:
        """Update the policy based on collected state-action data.

        Args:
            state_actions: The state-action pairs with evaluation metrics
            checkpoint: Whether to save a checkpoint of the updated actions

        Returns:
            Updated actions (system prompt, model parameters)
        """
        logger.info("Updating policy based on collected data")
        updated_actions = self.policy_updater.update(
            state_actions=state_actions,
            checkpoint=checkpoint,
        )
        logger.info("Policy update completed")
        return updated_actions

    def validate_policy(
        self,
        evaluator: Callable,
        evaluator_names: List[str],
        model: OpenAIModel,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        updated_actions: Optional[Actions] = None,
        run_id: Optional[str] = None,
        upsert: bool = False,
        limit: int = 100,
    ) -> StateActions:
        """Validate the policy using emulation.

        Args:
            state_actions: StateActions containing samples to test
            updated_actions: Updated actions to validate
            run_id: Optional run ID for tracking

        Returns:
            Results from the emulation
        """
        logger.info(f"Collecting state actions from {start_time} to {end_time}")
        state_actions = self.environment.data_collector.collect_data(
            start_date=start_time,
            end_date=end_time,
            evaluator_names=evaluator_names,
            limit=limit,
        )

        logger.info(f"Validating policy on {len(state_actions.samples)} samples")

        # Update the actions in the state_actions object
        if updated_actions:
            state_actions.actions = updated_actions

        # Run llm_runs emulation on the state_actions object
        logger.info(
            f"Running llm_runs emulation for {len(state_actions.samples)} samples"
        )
        self.environment.emulate_llm_call(state_actions, run_id=run_id)

        # sleep to allow for results to be uploaded to Arize
        sleep_duration = 30
        logger.info(
            f"Sleeping for {sleep_duration} seconds to allow for results to be uploaded to Arize"
        )
        time.sleep(sleep_duration)

        # Run evaluations on the state_actions object
        for evaluator_name in evaluator_names:
            logger.info(f"Running evaluation for {evaluator_name}")
            self.environment.emulate_eval(
                state_actions=state_actions,
                evaluator=evaluator,
                model=model,
                evaluator_name=evaluator_name,
                start_time=start_time,
                end_time=end_time,
                run_id=run_id,
                limit=limit,
            )

        logger.info("Policy validation completed")
        return state_actions

    def run_complete_workflow(
        self,
        start_date: datetime,
        end_date: datetime,
        evaluator_names: List[str],
        model: OpenAIModel,
        evaluator: Callable,
        limit: int = 100,
        checkpoint: bool = True,
        verbose: bool = True,
        upsert: bool = False,
    ) -> Dict[str, Any]:
        """Run the complete workflow from baseline validation to updated policy validation.

        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            evaluator_names: Names of evaluators to include
            model: OpenAI model to use
            evaluator: Callable evaluator to use
            limit: Maximum number of samples to collect
            checkpoint: Whether to save a checkpoint of the updated actions
            upsert: Whether to upsert the results to Arize

        Returns:
            Dictionary containing results from each step
        """

        logger.info(f"Starting complete workflow from {start_date} to {end_date}")

        # Step 1: Validate baseline
        baseline_id = f"baseline_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        baseline_state_actions = self.validate_policy(
            evaluator=evaluator,
            evaluator_names=evaluator_names,
            model=model,
            start_time=start_date,
            end_time=end_date,
            run_id=baseline_id,
            limit=limit,
        )

        # Step 2: Update policy
        updated_actions = self.update_policy(
            state_actions=baseline_state_actions,
            checkpoint=checkpoint,
        )

        if verbose:
            logger.info(
                f"Updated actions:\n{updated_actions.model_dump_json(indent=4)}"
            )

        # Step 3: Validate updated policy
        updated_id = f"updated_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.validate_policy(
            evaluator=evaluator,
            evaluator_names=evaluator_names,
            model=model,
            start_time=start_date,
            end_time=None,
            updated_actions=updated_actions,
            run_id=updated_id,
            limit=limit,
            upsert=upsert,
        )

        # # Return results from all steps
        # workflow_results = {
        #     "baseline_state_actions": baseline_state_actions,
        #     "updated_actions": updated_actions,
        #     "updated_validation_results": updated_results,
        # }
        workflow_results = {
            "state": "complete",
        }

        logger.info("Complete workflow completed successfully")
        return workflow_results
