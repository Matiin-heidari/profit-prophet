# exports the name of the training algorithm
from pathlib import Path

import numpy as np
from gymnasium import spaces
from negmas.outcomes import Outcome
from scml.oneshot.rl.observation import FlexibleObservationManager
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.context import GeneralContext, StrongSupplierContext, BalancedSupplierContext, WeakSupplierContext, StrongConsumerContext, BalancedConsumerContext, WeakConsumerContext
from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm

TrainingAlgorithm: type[BaseAlgorithm] = A2C
"""The algorithm used for training. You can use any stable_baselines3 algorithm or develop your own"""

MODEL_PATH = Path(__file__).parent / "models" / "mymodel"
"""The path in which train.py saves the trained model and from which myagent.py loads it"""

CONTEXTS = [
        "StrongSupplierContext",
        "BalancedSupplierContext",
        "WeakSupplierContext",
        "StrongConsumerContext",
        "BalancedConsumerContext",
        "WeakConsumerContext"
    ]
"""
The possible contexts for which we train the models.
For example StrongSupplierContext is a context in which the amount of consumers
is high in comparison to the number of competitors.
"""


def make_context(context_name: str) -> GeneralContext:
    """Generates a context based on the given string"""
    match context_name:
        case "StrongSupplierContext": return StrongSupplierContext()
        case "BalancedSupplierContext": return BalancedSupplierContext()
        case "WeakSupplierContext": return WeakSupplierContext()
        case "StrongConsumerContext": return StrongConsumerContext()
        case "BalancedConsumerContext": return BalancedConsumerContext()
        case "WeakConsumerContext": return WeakConsumerContext()
        case _: return GeneralContext()
    
    


class MyObservationManager(FlexibleObservationManager):
    """This is my observation manager implementing encoding and decoding the state used by the RL algorithm"""

    def make_space(self) -> spaces.MultiDiscrete | spaces.Box:
        """Creates the observation space"""
        return super().make_space()

    def encode(self, awi: OneShotAWI) -> np.ndarray:
        """Encodes an observation from the agent's state"""
        return super().encode(awi)

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return super().make_first_observation(awi)

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, Outcome | None]:
        """Gets the offers from an encoded state"""
        return super().get_offers(awi, encoded)


# ensure that the folder containing models is created
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
