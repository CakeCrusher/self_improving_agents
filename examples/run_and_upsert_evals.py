import os
from datetime import datetime, timedelta

from arize.exporter import ArizeExportClient
from arize.pandas.logger import Client
from arize.utils.types import Environments
from dotenv import load_dotenv
from phoenix.evals import OpenAIModel, llm_classify

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
primary_df.to_json("primary_df.json", orient="records")

print(primary_df)

# Run evals


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = OpenAIModel(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=OPENAI_API_KEY,
)

# load texts from original.md scale1.md scale3.md scale5.md
with open("original.md", "r") as f:
    EXAMPLE_ORIGINAL_TEXT = f.read()
with open("scale1.md", "r") as f:
    EXAMPLE_VARIATION_SCALE_1 = f.read()
with open("scale3.md", "r") as f:
    EXAMPLE_VARIATION_SCALE_3 = f.read()
with open("scale5.md", "r") as f:
    EXAMPLE_VARIATION_SCALE_5 = f.read()

EVAL_TEMPLATE = (
    f"""
You are tasked with evaluating formatting consistency between an original text and a rewritten version. You will be provided with an original text and the rewritten version. Your goal is to determine the formatting consistency score (scale of 1-5) of the new rewritten version compared to the original.

Use the following as examples on how this task is to be done, they are all based on the same "Original Text":

---

### Example Original Text:

```
{EXAMPLE_ORIGINAL_TEXT}
```

---

### Example Variation (Graded score 1, Completely Different Formatting):

```
{EXAMPLE_VARIATION_SCALE_1}
```

Formatting Consistency Score: 1

Reasoning:
The formatting completely differs, using alternative structures, headings, lists, and organization compared to the original.

---

### Example Variation (Graded score 3, Similar but Varied Formatting):

```
{EXAMPLE_VARIATION_SCALE_3}
```

Formatting Consistency Score: 3

Reasoning:
The formatting is similar but contains noticeable inconsistencies, such as varied heading styles, subheading differences, or minor structural variations.

---

### Example Variation (Graded score 5, Exactly Matching Formatting):

```
{EXAMPLE_VARIATION_SCALE_5}
```

Formatting Consistency Score: 5

Reasoning:
The formatting exactly matches the original, including headings, subheadings, block quotes, lists, and overall structural organization.

---
"""
    + """
Now, I will provide you with the user-provided original version and its rewritten version.

---

### Original Text:

```
{ORIGINAL_TEXT}
```

---

### New Rewritten Version:

```
{REWRITTEN_TEXT}
```

---

### Task:

Provide a formatting consistency score (1-5) for the new rewritten version.
"""
)


primary_df["ORIGINAL_TEXT"] = primary_df["attributes.llm.input_messages"].apply(
    lambda messages: messages[1]["message.content"]
)

primary_df["REWRITTEN_TEXT"] = primary_df["attributes.llm.output_messages"].apply(
    lambda messages: messages[0]["message.content"]
)

print(primary_df["ORIGINAL_TEXT"])
print(primary_df["REWRITTEN_TEXT"])

evals_df = llm_classify(
    dataframe=primary_df,
    model=model,
    template=EVAL_TEMPLATE,
    rails=["1", "2", "3", "4", "5"],
    provide_explanation=True,
)

print(evals_df)

# Create evaluation data for Arize upsert
print("\n\nSTARTING UPSERT EVALS TO ARIZE")


evals_df["eval.formatting_consistency_out_of_5.label"] = evals_df["label"]
evals_df["eval.formatting_consistency_out_of_5.explanation"] = evals_df["explanation"]

ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_MODEL_ID = os.getenv("ARIZE_MODEL_ID")

client = Client(
    space_id=ARIZE_SPACE_ID,
    developer_key=ARIZE_DEVELOPER_KEY,
    api_key=ARIZE_API_KEY,
)

evals_df["context.span_id"] = primary_df["context.span_id"]

# save evals_df to json
evals_df.to_json("evals_df.json", orient="records")

client.log_evaluations_sync(evals_df, ARIZE_MODEL_ID)
