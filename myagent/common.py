# exports the name of the training algorithm
import os
from pathlib import Path

import numpy as np
from gymnasium import spaces
from negmas.outcomes import Outcome
from scml.oneshot.rl.observation import FlexibleObservationManager
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.context import (
    GeneralContext,
    StrongSupplierContext,
    BalancedSupplierContext,
    WeakSupplierContext,
    StrongConsumerContext,
    BalancedConsumerContext,
    WeakConsumerContext,
)
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

TrainingAlgorithm: type[PPO] = PPO
"""The algorithm used for training. You can use any stable_baselines3 algorithm or develop your own"""

_model_dir = os.environ.get("MODEL_DIR", "").strip()
MODEL_PATH = (
    Path(_model_dir) if _model_dir else Path(__file__).parent / "models"
) / "mymodel"
"""The path in which train.py saves the trained model and from which myagent.py loads it.

The MODEL_DIR env var overrides the directory (resolved from the CWD if
relative), so alternative model sets (e.g. candidate_models/flex vs
candidate_models/accept) can be benchmarked without ever moving files in and
out of myagent/models. Unset = the canonical myagent/models (the only mode
that exists in a submission)."""

LOG_ROOT = "log"
"""Parent directory (relative to the run's CWD) for all log subfolders:
tensorboard_logs/, reward_component_logs/, context_usage_logs/."""

ALL_CONTEXTS = [
    "StrongSupplierContext",
    "BalancedSupplierContext",
    "WeakSupplierContext",
    "StrongConsumerContext",
    "BalancedConsumerContext",
    "WeakConsumerContext",
]
"""All contexts supported by the agent."""

CONTEXTS = [
    context.strip()
    for context in os.environ.get("TRAIN_CONTEXTS", ",".join(ALL_CONTEXTS)).split(",")
    if context.strip()
]
"""Contexts used for training. Can be overridden with TRAIN_CONTEXTS."""


def make_context(context_name: str) -> GeneralContext:
    """Create a context from its name."""
    match context_name:
        case "StrongSupplierContext":
            return StrongSupplierContext()
        case "BalancedSupplierContext":
            return BalancedSupplierContext()
        case "WeakSupplierContext":
            return WeakSupplierContext()
        case "StrongConsumerContext":
            return StrongConsumerContext()
        case "BalancedConsumerContext":
            return BalancedConsumerContext()
        case "WeakConsumerContext":
            return WeakConsumerContext()
        case _:
            return GeneralContext()


def get_parallelization_params(n_models_parallel: int = 1) -> dict:
    """Choose parallel env count based on local or Slurm CPU allocation."""
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
    """Observation manager with additional SCML-specific features."""

    def make_space(self) -> spaces.MultiDiscrete | spaces.Box:
        """Create the observation space."""
        base = super().make_space()
        n_extra = 6

        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(base.shape[0] + n_extra,),
            dtype=np.float32,
        )

    def encode(self, awi: OneShotAWI) -> np.ndarray:
        """Encode the agent state."""
        try:
            base = super().encode(awi)
        except ValueError as e:
            if "min() arg is an empty sequence" in str(e):
                space = self.make_space()

                if isinstance(space, spaces.MultiDiscrete):
                    return np.zeros_like(space.nvec, dtype=np.int64)

                if isinstance(space, spaces.Box):
                    return np.zeros(space.shape, dtype=space.dtype)

            raise

        input_price = max(
            1.0,
            awi.trading_prices[awi.my_input_product],
        )

        output_price = max(
            1.0,
            awi.trading_prices[awi.my_output_product],
        )

        production_cost = float(awi.profile.cost)

        # Cost features
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

        # Quantity pressure features
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

        # Margin feature
        margin = (
            output_price
            - input_price
            - production_cost
        ) / output_price

        margin_ratio = np.clip(
            (margin + 1.0) / 2.0,
            0.0,
            1.0,
        )

        extras = np.array(
            [
                disposal_cost_ratio,
                shortfall_penalty_ratio,
                production_cost_ratio,
                needed_supplies_ratio,
                needed_sales_ratio,
                margin_ratio,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([base, extras])

        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

        return obs

    def make_first_observation(self, awi: OneShotAWI) -> np.ndarray:
        """Create the first observation."""
        return self.encode(awi)

    def get_offers(
        self, awi: OneShotAWI, encoded: np.ndarray
    ) -> dict[str, Outcome | None]:
        """Decode offers from an encoded state."""
        return super().get_offers(awi, encoded)


# Create model directory.
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
