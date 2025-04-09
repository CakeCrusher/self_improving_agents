#!/usr/bin/env python
"""Example script demonstrating policy updating using LLM.

This script shows how to use the LLMPolicyUpdater to analyze state-action data
and generate updates to the system prompt and model parameters.
"""
import logging
from datetime import datetime

from dotenv import load_dotenv

from self_improving_agents.evaluator_handler.tracker import EvaluatorTracker
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

start_date = datetime.fromtimestamp(1743826307473 / 1000)

data = data_collector.collect_data(
    start_date=start_date,
    end_date=datetime.now(),
    evaluator_names=["formatting_consistency_out_of_5"],
)

# save to data_collection.json
with open("data_collection.json", "w") as f:
    f.write(data.model_dump_json(indent=4))
