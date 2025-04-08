"""Environment module for emulating and tracking LLM calls and evaluations.

This module provides components for emulating LLM calls and evaluations in a
controlled environment, with instrumentation for tracking and analysis.

Classes:
    LLMEnvironment: Environment for emulating and tracking LLM calls.
    EnvironmentSnapshot: Tracks the state of the environment during LLM calls.
"""

from .llm_environment import LLMEnvironment
from .snapshot import EnvironmentSnapshot

__all__ = ["LLMEnvironment", "EnvironmentSnapshot"]
