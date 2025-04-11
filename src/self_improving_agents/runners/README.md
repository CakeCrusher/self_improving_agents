# Runners

This directory contains components for orchestrating the evaluation and optimization loops of the self-improving agents system.

> Warining: add `.sia` to `.gitignore` your gitignore as it will contain your `space_id` and `model_id`

## WorkflowOrchestrator

The `WorkflowOrchestrator` class streamlines the self-improvement workflow by tying together the evaluation pipeline, environment emulation, and policy updating components.

### Key Features

- **Single entry point**: Provides a clean, high-level API for running the complete self-improvement workflow
- **Component integration**: Coordinates the interactions between evaluator handlers, environment, and policy components
- **Workflow steps**: Supports running the full workflow or individual steps as needed
- **Logging**: Comprehensive logging throughout the workflow

### Workflow Stages

The self-improvement workflow consists of these main stages:

1. **Run Evaluation Pipeline**: Transition existing evaluations to use `eval_pipeline` to store evaluation information
2. **Validate Baseline**: Analyze traces within a time window to establish baseline performance
3. **Update Policy**: Run policy updates based on collected state-action data
4. **Validate Updated Policy**: Test the updated policy actions using environment emulation

### Usage

#### Complete Workflow

```python
from datetime import datetime, timedelta
from self_improving_agents.runners import WorkflowOrchestrator

# Initialize the orchestrator (assumes environment variables are set)
orchestrator = WorkflowOrchestrator()

# Define time window
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

# Define evaluators to include
evaluator_names = ["correctness", "helpfulness"]

# Run the complete workflow
results = orchestrator.run_complete_workflow(
    start_date=start_date,
    end_date=end_date,
    evaluator_names=evaluator_names,
    limit=50,
    checkpoint=True,
)

# Access results
baseline_samples = results["baseline_state_actions"].samples
updated_actions = results["updated_actions"]
validation_results = results["updated_validation_results"]
```

#### Individual Components

You can also run individual components of the workflow:

```python
# Run just the baseline validation
baseline_state_actions = orchestrator.validate_baseline(
    start_date=start_date,
    end_date=end_date,
    evaluator_names=evaluator_names,
)

# Run just the policy update
updated_actions = orchestrator.update_policy(
    state_actions=baseline_state_actions,
)

# Run just the updated policy validation
results = orchestrator.validate_updated_policy(
    state_actions=baseline_state_actions,
    updated_actions=updated_actions,
)
```

#### Evaluation Pipeline

To run just the evaluation pipeline for a specific evaluator:

```python
from your_module import your_evaluator_function

orchestrator.run_eval_pipeline(
    evaluator=your_evaluator_function,
    evaluator_name="your_evaluator_name",
    evaluator_kwargs={
        # Your evaluator-specific parameters
        "threshold": 0.5,
    },
    get_telemetry_kwargs={
        "model_id": "your_model_id",
        "start_time": start_date.isoformat(),
        "end_time": end_date.isoformat(),
    },
    upsert=True,  # Set to True to upload results to Arize
)
```

### Prerequisites

Before using the `WorkflowOrchestrator`, ensure:

1. Required environment variables are configured:
   - `ARIZE_SPACE_ID`
   - `ARIZE_API_KEY`
   - `ARIZE_MODEL_ID`
   - `OPENAI_API_KEY`

2. You are saving traces on Arize with consistent attributes

3. You are using evaluations with consistent attributes

4. Your evaluators are compatible with the evaluation pipeline

See the `examples/workflow_example.py` file for a complete working example.
