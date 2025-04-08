#!/usr/bin/env python
"""Example script demonstrating the LLM environment.

This script shows how to use the LLM environment to emulate LLM calls
with instrumentation and tracking.
"""
import logging
from datetime import datetime

from dotenv import load_dotenv

from self_improving_agents.environment import LLMEnvironment

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""Run the LLM environment example."""
# Initialize the LLM environment
logger.info("Initializing LLM environment")
env = LLMEnvironment()

# Try to collect updated state actions from Arize
# try:
logger.info("Attempting to collect samples and updated actions from Arize")
# Set the date range
end_date = datetime.now()
start_date = datetime.fromtimestamp(1743826307473 / 1000)


state_actions = env.collect_updated_state_actions(
    start_date=start_date,
    end_date=end_date,
    evaluator_names=["formatting_classify"],  # Replace with your evaluator names
    limit=3,
)

logger.info(f"Collected {len(state_actions.samples)} samples with updated actions")

# Emulate LLM calls using the collected state_actions
# if state_actions.samples:
logger.info("Emulating LLM calls with collected state actions")
response = env.emulate_llm_call(
    state_actions=state_actions,
    run_id=f"example_run_{datetime.now().strftime('%Y%m%d%H%M%S')}",
)

logger.info(f"Response: {response['content']}")
logger.info(f"Tokens used: {response['usage']}")
logger.info(f"Duration: {response['duration']} seconds")
# else:
#     logger.warning("No samples collected, skipping LLM call emulation")
