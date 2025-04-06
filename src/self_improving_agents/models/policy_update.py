from typing import List

from pydantic import BaseModel, Field

from .state_action import Actions


class PolicyUpdate(BaseModel):
    """A policy update on its actions."""

    thoughts: List[str] = Field(
        description="A list of thoughts about the policy update."
    )
    actions: Actions = Field(description="The actions to update the policy with.")
