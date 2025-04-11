"""LLM-based policy updater for determining optimal action updates.

This module implements a policy updater that uses large language models to analyze
state-action-evaluation data and determine optimal updates to the action space.
"""
import itertools
import json
import logging
from typing import Dict, List, Optional

from openai import OpenAI

from self_improving_agents.models.policy_update import PolicyUpdate

from ..models.state_action import Actions, EvalConstant, Sample, StateActions
from .base import BasePolicy

logger = logging.getLogger(__name__)


class LLMPolicyUpdater(BasePolicy):
    """Policy updater that uses LLM to determine optimal updates to actions."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """Initialize the LLM policy updater.

        Args:
            client: OpenAI client (will create one if not provided)
            model: The model to use for policy updates
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generated responses
        """
        self.client = client or OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def update(self, state_actions: StateActions, checkpoint: bool = True) -> Actions:
        """Update the policy based on collected state-action data.

        Args:
            state_actions: The state-action pairs with evaluation metrics

        Returns:
            Updated actions (system prompt, model parameters)
        """
        messages = self._compose_update_messages(state_actions)
        # save the prompt to composed_prompt.txt
        with open("composed_prompt.txt", "w") as f:
            f.write(messages[1]["content"])
        policy_update = self._get_llm_update(messages)

        try:
            # Parse the response into the updated Actions
            updated_actions = self._parse_update_response(
                policy_update.actions, state_actions.actions
            )
            if checkpoint:
                self.save_checkpoint(updated_actions)
            return updated_actions
        except Exception as e:
            logger.error(f"Failed to parse LLM update response: {e}")
            # Return the original actions if parsing fails
            return state_actions.actions

    def _compose_update_messages(
        self, state_actions: StateActions, samples_max_character_length: int = 16000
    ) -> List[Dict[str, str]]:
        """Compose the prompt for the LLM to update the policy.

        Args:
            state_actions: State-action pairs with evaluations

        Returns:
            Structured prompt for the LLM
        """
        # System message with instructions
        # TODO: update this to eliminate constraints on action space i.e. "system prompts and model parameters"
        system_message = {
            "role": "system",
            "content": """You are an AI policy optimizer that analyzes interaction data and evaluation metrics
            to improve system prompts and model parameters. Your goal is to suggest updates to the
            current policy (system prompt and model parameters) based on the performance data provided.

            Analyze the samples, their evaluation metrics, and determine what changes should be made to
            the system prompt and model parameters to improve performance based on the evaluation criteria.
            """,
        }

        # Create a summary of the evaluation results
        formatted_samples = [
            self._format_sample(sample, i)
            for i, sample in enumerate(state_actions.samples)
        ]
        # Truncate samples to stay under max character length
        formatted_samples = list(
            itertools.takewhile(
                lambda x: sum(
                    len(s) for s in formatted_samples[: formatted_samples.index(x) + 1]
                )
                <= samples_max_character_length,
                formatted_samples,
            )
        )
        formatted_samples_str = "\n".join(formatted_samples)

        # User message with structured data
        prompt = f"""
# Current Policy Configuration
- System Prompt: <SYSTEM_PROMPT>{state_actions.actions.system_prompt}</SYSTEM_PROMPT>
- Model name: <MODEL_NAME>{state_actions.actions.model}</MODEL_NAME>

# Evaluation Constants
<EVALUATION_CONSTANTS>
{self._format_eval_constants(state_actions.eval_constants)}
</EVALUATION_CONSTANTS>

# Evaluation Summary
<EVALUATION_SUMMARY>
{self._create_evaluation_summary(state_actions.samples)}
</EVALUATION_SUMMARY>

# Sample Examples (showing {len(formatted_samples)} of {len(state_actions.samples)})
<SAMPLE_EXAMPLES>
{formatted_samples_str}
</SAMPLE_EXAMPLES>

Based on this data, please suggest specific updates to the system prompt and model parameters
that would improve performance according to the evaluation metrics.
"""

        return [system_message, {"role": "user", "content": prompt}]

    def _create_evaluation_summary(self, samples: List[Sample]) -> str:
        """Create a summary of evaluation results across all samples.

        Args:
            samples: List of samples with evaluations

        Returns:
            Formatted evaluation summary
        """
        if not samples:
            return "No samples available for evaluation."

        # Get unique evaluator names
        eval_names = set()
        for sample in samples:
            for eval_metric in sample.evals:
                eval_names.add(eval_metric.name)

        summary = []
        for eval_name in eval_names:
            # Collect all scores for this evaluator
            scores: List[float] = []
            for sample in samples:
                for eval_metric in sample.evals:
                    if (
                        eval_metric.name == eval_name
                        and eval_metric.eval_score is not None
                    ):
                        scores.append(eval_metric.eval_score)

            # Calculate statistics
            if scores:
                avg_score: float | str = "N/A"
                try:
                    avg_score = sum([float(s) for s in scores]) / len(scores)
                except Exception as e:
                    logger.error(
                        f"Failed to calculate average score for {eval_name}: {e}"
                    )
                    avg_score = "N/A"
                summary.append(
                    f"- {eval_name}: Average score: {avg_score}, Samples: {len(scores)}"
                )

        return "\n".join(summary)

    def _format_eval_constants(self, eval_constants: List[EvalConstant]) -> str:
        """Format evaluation constants for the prompt.

        Args:
            eval_constants: List of evaluation constants

        Returns:
            Formatted evaluation constants string
        """
        result = []
        for ec in eval_constants:
            result.append(
                f"""## {ec.name}
- Template: <EVAL_TEMPLATE>{ec.eval_template}</EVAL_TEMPLATE>
- Rails: <EVAL_RAILS>{json.dumps(ec.eval_rails, indent=2)}</EVAL_RAILS>"""
            )

        return "\n".join(result)

    def _format_sample(
        self, sample: Sample, index: int, chat_history_max_length: int = 3
    ) -> str:
        """Format a single sample example for the prompt.

        Args:
            sample: Sample example to format
            chat_history_max_length: Maximum number of chat history messages to show
        Returns:
            Formatted sample string
        """
        chat_history_to_show = sample.chat_history[
            min(1, len(sample.chat_history) - chat_history_max_length) :
        ]
        evals_to_show = "\n".join(
            [
                f"- {eval_metric.name}: Score {eval_metric.eval_score}\n  Reasoning: <EVAL_REASONING>{eval_metric.eval_reasoning}</EVAL_REASONING>"
                for eval_metric in sample.evals
            ]
        )
        return f"""<SAMPLE_{index}>
### Chat History
<CHAT_HISTORY>
{chat_history_to_show}
</CHAT_HISTORY>

### Output Generation
<OUTPUT_GENERATION>
{sample.output_generation}
</OUTPUT_GENERATION>

### Evaluations
<EVALUATIONS>
{evals_to_show}
</EVALUATIONS>
</SAMPLE_{index}>"""

    # TODO: this should be producing diffs
    def _get_llm_update(self, messages: List[Dict[str, str]]) -> PolicyUpdate:
        """Get the LLM's suggested updates based on the prompt.

        Args:
            prompt: The structured prompt to send to the LLM

        Returns:
            The LLM's response with suggested updates
        """

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=PolicyUpdate,
        )

        logger.info(
            "UPDATE THOUGHTS: ",
            json.dumps(response.choices[0].message.parsed.thoughts, indent=2),
        )

        # This garbage is needed because of mypy
        policy_update = PolicyUpdate(**response.choices[0].message.parsed.model_dump())

        return policy_update

    # TODO: This will play a role when we do diffs
    def _parse_update_response(
        self, updated_actions: Actions, current_actions: Actions
    ) -> Actions:
        """Parse the LLM response into an updated Actions object.

        Args:
            response: JSON string response from the LLM
            current_actions: Current actions to update

        Returns:
            Updated Actions object
        """
        try:
            # Create and return the updated Actions
            return updated_actions
        except Exception as e:
            logger.error(f"Failed to parse LLM update response: {e}")
            # Return the original actions if parsing fails
            return current_actions
