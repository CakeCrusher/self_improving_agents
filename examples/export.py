import os
from datetime import datetime, timedelta

from arize.exporter import ArizeExportClient
from arize.utils.types import Environments
from dotenv import load_dotenv

load_dotenv()
print("\n\nSTARTING EVALUATION RUN")

ARIZE_DEVELOPER_KEY = os.getenv("ARIZE_DEVELOPER_KEY")
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")
ARIZE_MODEL_ID = os.getenv("ARIZE_MODEL_ID")

now = datetime.now()


# Exporting your dataset into a dataframe
client = ArizeExportClient(api_key=ARIZE_DEVELOPER_KEY)
primary_df = client.export_model_to_df(
    model_id=ARIZE_MODEL_ID,
    space_id=ARIZE_SPACE_ID,
    environment=Environments.TRACING,
    start_time=now - timedelta(days=2),
    end_time=now,
)
# save primary_df to json
primary_df.head(3).to_json("primary_df.json", orient="records", indent=4)
