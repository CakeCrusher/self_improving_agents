# Initialize arize platform tracing
# Import open-telemetry dependencies
import os

from arize.otel import register
from dotenv import load_dotenv

load_dotenv()

ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_MODEL_ID = os.getenv("ARIZE_MODEL_ID")

print("\n\nREGISTERING TRACER PROVIDER")
# Setup OTel via our convenience function
tracer_provider = register(
    space_id=ARIZE_SPACE_ID,  # in app space settings page
    api_key=ARIZE_API_KEY,  # in app space settings page
    project_name=ARIZE_MODEL_ID,  # name this to whatever you would like
)
