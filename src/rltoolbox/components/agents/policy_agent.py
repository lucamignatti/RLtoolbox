import numpy as np
import random
from typing import Dict, Any

from ....core.base import RLComponent


class PolicyAgent(RLComponent):
    """
    Policy-based agent that selects actions from action probabilities.

    Expects action probabilities to be provided in context by another component.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_actions = config.get("num_actions", 2)
        self.deterministic = config.get("deterministic", False)

    def action_selection(self, context: Dict[str, Any]) -> None:
        """Select action based on policy probabilities."""
        if "action_probs" not in context:
            # Fallback to random if no probabilities available
            action = random.randint(0, self.num_actions - 1)
        else:
            action_probs = np.array(context["action_probs"])

            if self.deterministic:
                # Select action with highest probability
                action = np.argmax(action_probs)
            else:
                # Sample from probability distribution
                action = np.random.choice(self.num_actions, p=action_probs)

        context["action"] = action

    def validate_config(self) -> bool:
        """Validate policy agent configuration."""
        return self.num_actions > 0
