import os
from datetime import datetime, timedelta

from arize.utils.types import Environments
from dotenv import load_dotenv
from phoenix.evals import OpenAIModel, llm_classify

from self_improving_agents.evaluator_handler.eval_pipeline import EvalPipeline
from self_improving_agents.evaluator_handler.evaluator_saver import EvaluatorSaver
from self_improving_agents.runners.arize_connector import ArizeConnector

load_dotenv()
print("\n\nSTARTING EVALUATION RUN")

ARIZE_DEVELOPER_KEY = os.getenv("ARIZE_DEVELOPER_KEY")
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")
ARIZE_MODEL_ID = os.getenv("ARIZE_MODEL_ID")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

### Original Chat Completions Input text:

```
{attributes.llm.input_messages}
```

---

### New Rewritten output in Chat Completions format:

```
{attributes.llm.output_messages}
```

---

### Task:

Provide a formatting consistency score (1-5) for the new rewritten version.
"""
)

evaluator_saver = EvaluatorSaver()
arize_connector = ArizeConnector(
    developer_key=ARIZE_DEVELOPER_KEY,
    space_id=ARIZE_SPACE_ID,
    model_id=ARIZE_MODEL_ID,
    api_key=ARIZE_API_KEY,
)

# Initialize pipeline
pipeline = EvalPipeline(
    evaluator_saver=evaluator_saver, arize_connector=arize_connector
)

model = OpenAIModel(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=OPENAI_API_KEY,
)

start_date = datetime.now() - timedelta(days=2)
end_date = datetime.now()

# Run evaluation
results = pipeline.run_pipeline(
    evaluator=llm_classify,
    evaluator_name="formatting_classify",
    evaluator_kwargs={
        "model": model,
        "template": EVAL_TEMPLATE,
        "rails": ["1", "2", "3", "4", "5"],
        "provide_explanation": True,
    },
    get_telemetry_kwargs={
        "model_id": ARIZE_MODEL_ID,
        "space_id": ARIZE_SPACE_ID,
        "environment": Environments.TRACING,
        "start_time": start_date,
        "end_time": end_date,
    },
    upsert=True,
)

print(f"Completed evaluation run with {results.shape[0]} rows")

# save results to json file
results.to_json("results.json", orient="records", indent=4)
