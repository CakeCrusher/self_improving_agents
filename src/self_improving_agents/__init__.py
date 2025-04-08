"""Self Improving Agents package.

This package provides tools and utilities for evaluating and optimizing
LLM agents. It includes components for data modeling,
evaluation, policy configuration and optimization, and orchestration.

Modules:
    models: Data structures and schemas using Pydantic.
    evaluators: Components for pulling and replicating evaluations.
    policy: Components for defining actions and updating the policy.
    runners: Orchestration of evaluation/optimization loops.
    utils: Shared utilities.
    instrumentation: Wrappers for tracking eval function usage.
    environment: Components for emulating LLM calls and evaluations.
"""

__version__ = "0.1.0"
