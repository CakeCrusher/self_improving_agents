# examples/update_policy.py
"""Example demonstrating the state-action collection and policy update workflow."""
from datetime import datetime

from dotenv import load_dotenv

from self_improving_agents.evaluator_handler.tracker import EvaluatorTracker
from self_improving_agents.runners.arize_connector import ArizeConnector
from self_improving_agents.runners.data_collection_runner import DataCollectionRunner

load_dotenv()


# Initialize components
evaluator_tracker = EvaluatorTracker()
arize_connector = ArizeConnector()

# Create runner
runner = DataCollectionRunner(
    evaluator_tracker=evaluator_tracker, arize_connector=arize_connector
)

# Set date range for data collection
start_date = datetime.fromtimestamp(
    1743826307473 / 1000
)  # Convert milliseconds to seconds

# Collect state-action pairs
state_actions_pair = runner.collect_data(
    start_date=start_date, evaluator_names=["formatting_classify"], limit=3
)

# Print summary of collected data
if state_actions_pair:
    # Example of accessing data
    first_pair = state_actions_pair
    print(f"Sample action: {first_pair.actions.model}")
    print(f"Sample eval template: {len(first_pair.eval_constants[0].eval_template)}")

    # In a real implementation, you would now:
    # 1. Process these state-action pairs
    # 2. Update your policy based on them
    # 3. Save the updated policy
    print("Now you would update your policy based on these state-action pairs")
else:
    print("Failed to collect")
