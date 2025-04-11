#!/usr/bin/env python
"""Example script for using the workflow orchestrator.

This script demonstrates how to use the WorkflowOrchestrator to run the complete
self-improvement workflow or individual components of it.
"""
import logging
from datetime import datetime, timedelta

from phoenix.evals import OpenAIModel, llm_classify

from self_improving_agents.runners.orchestrator import WorkflowOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


"""Run the example workflow."""
# Initialize the orchestrator
# Note: This assumes the necessary environment variables are set
# ARIZE_SPACE_ID, ARIZE_API_KEY, ARIZE_MODEL_ID, OPENAI_API_KEY
orchestrator = WorkflowOrchestrator()

# Define time window for data collection (last 7 days)
# transform this to datetime 4/10/2025, 09:39 PM 10.7s
end_date = datetime.fromisoformat("2025-04-10T21:42:10.700")
print(f"End date: {end_date} {type(end_date)}")
start_date = end_date - timedelta(days=7)
print(f"Start date: {start_date} {type(start_date)}")
# Define evaluator names to include
evaluator_names = [
    "formatting_classify",
    # Add any other evaluators you're using
]

model = OpenAIModel(model="gpt-4o-mini")

# Example 1: Run the complete workflow
logger.info("Running complete workflow example")
results = orchestrator.run_complete_workflow(
    start_date=start_date,
    end_date=end_date,
    evaluator_names=evaluator_names,
    model=model,
    evaluator=llm_classify,
    limit=3,  # Adjust based on your needs
    checkpoint=True,
    upsert=True,
    verbose=True,
)

# Print some results
logger.info("Workflow results: %s", results)
# Example 2: Run individual components
# Uncomment the following section if you want to run components separately

"""
# Step 1: Validate baseline
logger.info("Running baseline validation only")
baseline_state_actions = orchestrator.validate_baseline(
    start_date=start_date,
    end_date=end_date,
    evaluator_names=evaluator_names,
    limit=50,
)

# Step 2: Update policy
logger.info("Running policy update only")
updated_actions = orchestrator.update_policy(
    state_actions=baseline_state_actions,
)

# Step 3: Validate updated policy
logger.info("Running updated policy validation only")
orchestrator.validate_updated_policy(
    state_actions=baseline_state_actions,
    updated_actions=updated_actions,
)
"""

# Example 3: Run just the eval pipeline for a specific evaluator
# Uncomment if you need to run the pipeline for a specific evaluator

"""
# This assumes you have an evaluator function defined
from your_module import your_evaluator_function

logger.info("Running eval pipeline for a specific evaluator")
orchestrator.run_eval_pipeline(
    evaluator=your_evaluator_function,
    evaluator_name="your_evaluator_name",
    evaluator_kwargs={
        # Your evaluator-specific parameters
        "threshold": 0.5,
    },
    get_telemetry_kwargs={
        "model_id": orchestrator.environment.arize_model_id,
        "start_time": start_date.isoformat(),
        "end_time": end_date.isoformat(),
    },
    upsert=True,  # Set to True to upload results to Arize
)
"""
