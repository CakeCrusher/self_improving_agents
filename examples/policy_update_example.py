#!/usr/bin/env python
"""Example script demonstrating policy updating using LLM.

This script shows how to use the LLMPolicyUpdater to analyze state-action data
and generate updates to the system prompt and model parameters.
"""
import logging
from datetime import datetime

from dotenv import load_dotenv

from self_improving_agents.evaluator_handler.tracker import EvaluatorTracker
from self_improving_agents.policy import LLMPolicyUpdater
from self_improving_agents.runners.arize_connector import ArizeConnector
from self_improving_agents.runners.data_collection_runner import DataCollectionRunner

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""Run the policy update example."""
# Initialize components
evaluator_tracker = EvaluatorTracker()
arize_connector = ArizeConnector()

# Initialize the data collection runner
data_collector = DataCollectionRunner(
    evaluator_tracker=evaluator_tracker,
    arize_connector=arize_connector,
)

# Initialize the policy updater
policy_updater = LLMPolicyUpdater(
    model="gpt-4o-mini",  # Or any model you prefer
    temperature=0.7,
)

# Collect data from the last day
end_date = datetime.now()
start_date = datetime.fromtimestamp(1743826307473 / 1000)
try:
    # Get the evaluator names you want to analyze
    evaluator_names = ["formatting_classify"]  # Replace with your evaluator names

    logger.info(f"Collecting data from {start_date} to {end_date}...")
    state_actions = data_collector.collect_data(
        start_date=start_date,
        end_date=end_date,
        evaluator_names=evaluator_names,
        limit=3,  # Adjust based on your needs
    )

    logger.info(f"Collected {len(state_actions.samples)} samples for analysis")

    # Update the policy based on collected data
    logger.info("Updating policy based on collected data...")
    updated_actions = policy_updater.update(state_actions)

    # Display the results
    logger.info("\n=== Original actions ===")
    logger.info(state_actions.actions.model_dump_json(indent=2))
    logger.info("\n=== Updated actions ===")
    logger.info(updated_actions.model_dump_json(indent=2))

except Exception as e:
    logger.error(f"Error in policy update example: {e}")
