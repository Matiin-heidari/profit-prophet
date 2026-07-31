# exports the name of the training algorithm
import os
from pathlib import Path
from typing import Callable

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

TrainingAlgorithm: type[BaseAlgorithm] = PPO
"""The algorithm used for training. You can use any stable_baselines3 algorithm or develop your own"""

_model_dir = os.environ.get("MODEL_DIR", "").strip()
MODEL_PATH = (
    Path(_model_dir) if _model_dir else Path(__file__).parent / "models"
) / "mymodel"
"""The path in which train.py saves the trained model and from which myagent.py loads it.

The MODEL_DIR env var overrides the directory (resolved from the CWD if
relative). Unset = the canonical myagent/models"""


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """Linear decay from ``initial_value`` to ``final_value`` over training.

    SB3 calls the returned callable with ``progress_remaining``, which goes
    from 1.0 (start) to 0.0 (end of training).
    """
    def schedule(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return schedule


def cosine_schedule(initial_value: float, final_value: float = 0.0):
    """Cosine decay from ``initial_value`` to ``final_value`` over training.

    Decays slowly at first and last, faster in the middle - often converges
    better than a linear ramp because it holds a high LR a bit longer before
    dropping.
    """
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return final_value + (initial_value - final_value) * cosine_decay

    return schedule


def get_lr_schedule() -> float | Callable[[float], float]:
    """Resolve the PPO learning-rate schedule from env vars.

    - ``LR_INITIAL``: starting learning rate (default ``3e-4``, SB3's PPO default).
    - ``LR_FINAL``: learning rate at the end of training (default ``1e-5``).
    - ``LR_SCHEDULE``: ``linear`` (default), ``cosine``, or ``constant``.

    Starting high and decaying lets the agent explore broadly early on, then
    take smaller, more stable update steps as it converges - so training
    doesn't overshoot a good policy once it's close to one.
    """
    initial = float(os.environ.get("LR_INITIAL", "3e-4"))
    final = float(os.environ.get("LR_FINAL", "1e-5"))
    schedule_type = os.environ.get("LR_SCHEDULE", "linear").lower()

    if schedule_type == "constant":
        return initial
    if schedule_type == "cosine":
        return cosine_schedule(initial, final)
    return linear_schedule(initial, final)

MODEL_PATH = Path(__file__).parent / "models" / "mymodel"
"""The path in which train.py saves the trained model and from which myagent.py loads it."""

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


def make_context(context_name: str, non_competitors=None) -> GeneralContext:
    """Create a context from its name.

    ``non_competitors`` (a tuple of agent classes) overrides the pool the
    context fills non-agent world slots from; None keeps the scml default
    (Greedy, RandDist, EqualDist). Only training envs pass a pool (see
    myagent/opponents.py) — eval and deployment always use the default so
    curves stay comparable.
    """
    kwargs = {} if non_competitors is None else {"non_competitors": tuple(non_competitors)}
    match context_name:
        case "StrongSupplierContext":
            return StrongSupplierContext(**kwargs)
        case "BalancedSupplierContext":
            return BalancedSupplierContext(**kwargs)
        case "WeakSupplierContext":
            return WeakSupplierContext(**kwargs)
        case "StrongConsumerContext":
            return StrongConsumerContext(**kwargs)
        case "BalancedConsumerContext":
            return BalancedConsumerContext(**kwargs)
        case "WeakConsumerContext":
            return WeakConsumerContext(**kwargs)
        case _:
            return GeneralContext(**kwargs)


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
