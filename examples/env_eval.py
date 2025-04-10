#!/usr/bin/env python
"""Example script demonstrating the LLM environment eval emulator.

This script demonstrates how to use the LLM environment to emulate an evaluation.
"""

import json
import logging
from datetime import datetime

from dotenv import load_dotenv
from phoenix.evals import OpenAIModel, llm_classify

from self_improving_agents.environment.llm_environment import LLMEnvironment
from self_improving_agents.environment.snapshot import EnvironmentSnapshot

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""Run the LLM environment example."""
# Initialize the LLM environment
logger.info("Initializing LLM environment")

env = LLMEnvironment()

model = OpenAIModel(model="gpt-4o-mini")

snapshot_config = EnvironmentSnapshot()
snapshot = snapshot_config.load()
print("snapshot: ", snapshot)


if not snapshot or not snapshot.start_time or not snapshot.end_time:
    raise ValueError("Snapshot not found")


start_datetime = datetime.fromisoformat(snapshot.start_time)
end_datetime = (
    datetime.fromisoformat(snapshot.end_time) if snapshot.end_time else datetime.now()
)

print("start_time: ", start_datetime)

collected_data = env.collect_updated_state_actions(
    start_date=start_datetime,
    end_date=end_datetime,
    evaluator_names=["formatting_classify"],
    limit=10,
)
logger.info(f"Collected {len(collected_data.samples)} samples")

eval_results = env.emulate_eval(
    state_actions=collected_data,
    model=model,
    evaluator=llm_classify,
    evaluator_name="formatting_classify",
    start_time=start_datetime,
    end_time=end_datetime,
    run_id="example_eval",
)

with open("eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=4)
