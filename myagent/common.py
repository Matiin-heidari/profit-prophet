# exports the name of the training algorithm
import os
from pathlib import Path

import numpy as np
from gymnasium import spaces
from negmas.outcomes import Outcome
from scml.oneshot.rl.observation import FlexibleObservationManager
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.context import GeneralContext, StrongSupplierContext, BalancedSupplierContext, WeakSupplierContext, StrongConsumerContext, BalancedConsumerContext, WeakConsumerContext
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

TrainingAlgorithm: type[BaseAlgorithm] = PPO
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
    
def get_parallelization_params(n_models_parallel: int = 1) -> dict:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    if slurm_cpus:
        total_cores = int(slurm_cpus)
        source = "SLURM_CPUS_PER_TASK"
    else:
        total_cores = os.cpu_count() or 1
        source = "os.cpu_count()"

    usable_cores = max(1, total_cores - 1)
    cores_per_model = max(1, usable_cores // n_models_parallel)

    # n_envs for the case of running on HPC
    n_envs = min(cores_per_model, 8)
    # in case of running on the local machine
    #n_envs = 2

    print(
        f"Detected {total_cores} cores from {source} → "
        f"using {n_envs} envs per model "
        f"({n_models_parallel} model(s) training in parallel)"
    )

    return {
        "n_envs": n_envs,
        "n_models_parallel": n_models_parallel,
    }


class MyObservationManager(FlexibleObservationManager):
    """This is my observation manager implementing encoding and decoding the state used by the RL algorithm"""
    def make_space(self) -> spaces.MultiDiscrete | spaces.Box:
        """Creates the observation space"""
        base = super().make_space()
        n_extra = 8

        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(base.shape[0] + n_extra,),
            dtype=np.float32,
        )


    def encode(self, awi):
        base = super().encode(awi)

        competitors_ratio = np.clip(
            awi.n_competitors / 20,
            0.0,
            1.0,
        )

        input_price = max(
            1.0,
            awi.trading_prices[awi.my_input_product],
        )

        output_price = max(
            1.0,
            awi.trading_prices[awi.my_output_product],
        )

        production_cost = float(awi.profile.cost)

        # Cost-related features
        disposal_cost_ratio = np.clip(
            awi.current_disposal_cost / (2.0 * input_price),
            0.0,
            1.0,
        )

        shortfall_penalty_ratio = np.clip(
            awi.current_shortfall_penalty / (2.0 * output_price),
            0.0,
            1.0,
        )

        production_cost_ratio = np.clip(
            production_cost / (2.0 * input_price),
            0.0,
            1.0,
        )

        # Urgency features
        needed_supplies_ratio = np.clip(
            (awi.needed_supplies / max(1, awi.n_lines) + 1.0) / 2.0,
            0.0,
            1.0,
        )

        needed_sales_ratio = np.clip(
            (awi.needed_sales / max(1, awi.n_lines) + 1.0) / 2.0,
            0.0,
            1.0,
        )

        price_increase_ratio = np.clip(
            output_price / max(1.0, input_price),
            0.0,
            2.0,
        ) / 2.0

        relative_time = np.clip(
            awi.current_step / max(1, awi.n_steps - 1),
            0.0,
            1.0,
        )

        extras = np.array(
            [
                competitors_ratio,
                disposal_cost_ratio,
                shortfall_penalty_ratio,
                production_cost_ratio,
                needed_supplies_ratio,
                needed_sales_ratio,
                relative_time,
                price_increase_ratio,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([base, extras])
        return obs

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Creates the initial observation (returned from gym's reset())"""
        return self.encode(awi) # to be consistent with the changed encode function

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, Outcome | None]:
        """Gets the offers from an encoded state"""
        return super().get_offers(awi, encoded)


# ensure that the folder containing models is created
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
