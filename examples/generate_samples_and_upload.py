# import from inputs.json
import json
import os
import random

from arize.otel import register
from dotenv import load_dotenv
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor

with open("inputs.json", "r") as file:
    inputs = json.load(file)


# Set random seed for reproducibility
random.seed(42)

# Shuffle the inputs list in place
random.shuffle(inputs)

# Initialize arize platform tracing
# Import open-telemetry dependencies


load_dotenv()

ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_MODEL_ID = os.getenv("ARIZE_MODEL_ID")
print(f"ARIZE_SPACE_ID: {ARIZE_SPACE_ID}")
print(f"ARIZE_API_KEY: {ARIZE_API_KEY}")
print(f"ARIZE_MODEL_ID: {ARIZE_MODEL_ID}")

print("\n\nREGISTERING TRACER PROVIDER")
# Setup OTel via our convenience function
tracer_provider = register(
    space_id=ARIZE_SPACE_ID,  # in app space settings page
    api_key=ARIZE_API_KEY,  # in app space settings page
    project_name=ARIZE_MODEL_ID,  # name this to whatever you would like
)

# Import the automatic instrumentor from OpenInference

print("\n\nINSTRUMENTING OPENAI")
# Finish automatic instrumentation
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Make api calls to openai
SYSTEM_PROMPT = """I want you to rewrite what the user provides such that it invokes different ways of saying the same concept an maybe with different metaphors, data, and ideas. Make sure that the formatting is completely different formatting."""


client = OpenAI()

print("\n\nSTARTING GENERATE SAMPLES")

limit = 3
for idx, input in enumerate(inputs[:limit]):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input},
        ],
    )
    print(f"\n\nGenerated {idx} of {limit}\n")
