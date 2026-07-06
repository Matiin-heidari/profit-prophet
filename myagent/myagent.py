"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML competition.
"""

import csv
import os
import random
from pathlib import Path

import numpy as np

from scml.oneshot.rl.action import FlexibleActionManager
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper

from .action import AcceptFlagActionManager
from .common import (
    ALL_CONTEXTS,
    LOG_ROOT,
    MODEL_PATH,
    MyObservationManager,
    TrainingAlgorithm,
    make_context,
)


def _matching_action_manager(context, model):
    """Pick the action manager whose space matches the model's saved one.

    Models trained before the AcceptFlag manager existed use the plain
    FlexibleActionManager space ([max_q+1, n_prices] per partner); AcceptFlag
    models have one extra quantity value ([max_q+2, ...]). Matching on the
    saved `model.action_space` makes deployment self-configuring, so mixed
    sets of old- and new-space models load correctly with no flags to keep
    in sync.
    """
    empty = np.zeros(0, dtype=np.int64)
    model_nvec = np.asarray(getattr(model.action_space, "nvec", empty))
    candidates = (AcceptFlagActionManager(context), FlexibleActionManager(context))
    for manager in candidates:
        space = manager.make_space()
        if np.array_equal(np.asarray(getattr(space, "nvec", empty)), model_nvec):
            return manager
    # No exact match (e.g. a model saved with different context params):
    # keep the historical default rather than failing the whole agent.
    return candidates[-1]


class MyAgent(OneShotRLAgent):
    """RL agent that loads one trained model per context."""

    def __init__(self, *args, **kwargs):
        base_name = MODEL_PATH.name
        self.paths: list[Path] = []

        observation_managers = []
        action_managers = []
        models = []

        # The runtime agent should always load all supported contexts.
        for context_name in ALL_CONTEXTS:
            model_path = MODEL_PATH.parent / f"{base_name}{context_name}"
            self.paths.append(model_path)

            model = TrainingAlgorithm.load(model_path, device="cpu")
            context = make_context(context_name)
            observation_managers.append(MyObservationManager(context, continuous=True))
            action_managers.append(_matching_action_manager(context, model))
            models.append(model_wrapper(model, deterministic=True))

        models = tuple(models)

        kwargs.update(
            dict(
                models=models,
                observation_managers=observation_managers,
                action_managers=action_managers,
            )
        )

        super().__init__(*args, **kwargs)

        # Each loaded model has a stale SB3 `.seed` attribute baked in from
        # training (see CLAUDE.md §5). SB3's load() -> _setup_model() calls
        # set_random_seed(self.seed) for every model, which resets Python's
        # global random/np.random to that fixed value on EVERY MyAgent
        # construction (i.e. every world it plays in) - collapsing what should
        # be independent per-world randomness across a whole tournament/shard.
        # Reseed from OS entropy now that loading is done.
        random.seed()
        np.random.seed()

    def init(self):
        super().init()
        self._log_context_usage()

    def context_switch(self) -> None:
        """Select the per-context model from this agent's position."""
        self._valid_index = self._select_index()
        if self.has_no_valid_model() and self._fallback_agent is None:
            self.setup_fallback()

    def _select_index(self) -> int:
        """Map the current AWI to an index into ALL_CONTEXTS, or -1 for fallback."""
        try:
            awi = self.awi
            first = bool(awi.is_first_level)
            # Supplier competes on selling (consumers); consumer on buying (suppliers).
            n_side = len(awi.my_consumers) if first else len(awi.my_suppliers)
            d = n_side - awi.n_competitors
            if d <= -1:
                strength = "Strong"
            elif d <= 1:
                strength = "Balanced"
            else:
                strength = "Weak"
            side = "Supplier" if first else "Consumer"
            target = f"{strength}{side}Context"
            return ALL_CONTEXTS.index(target) if target in ALL_CONTEXTS else -1
        except Exception:
            return -1

    def _log_context_usage(self) -> None:
        """Record which per-context model (or heuristic fallback) was selected
        for this world. One row per world. Robust and side-effect-safe: any
        failure (e.g. read-only filesystem in a submission) is swallowed, and it
        can be disabled entirely with LOG_CONTEXT_USAGE=0."""
        if os.environ.get("LOG_CONTEXT_USAGE", "1") == "0":
            return
        try:
            idx = getattr(self, "_valid_index", -1)
            chosen = ALL_CONTEXTS[idx] if 0 <= idx < len(ALL_CONTEXTS) else "fallback"
            awi = self.awi
            n_suppliers = len(getattr(awi, "my_suppliers", []) or [])
            n_consumers = len(getattr(awi, "my_consumers", []) or [])
            n_competitors = getattr(awi, "n_competitors", "")
            first = bool(getattr(awi, "is_first_level", False))
            n_side = n_consumers if first else n_suppliers
            d = n_side - n_competitors if isinstance(n_competitors, int) else ""
            run = os.environ.get("RUN_NAME", "default")
            log_dir = Path(LOG_ROOT) / "context_usage_logs" / run
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / f"usage_{os.getpid()}.csv"
            is_new = not path.exists()
            with open(path, "a", newline="") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(
                        [
                            "agent_id",
                            "chosen",
                            "level",
                            "n_suppliers",
                            "n_consumers",
                            "n_competitors",
                            "d",
                        ]
                    )
                writer.writerow(
                    [
                        self.id,
                        chosen,
                        getattr(awi, "level", ""),
                        n_suppliers,
                        n_consumers,
                        n_competitors,
                        d,
                    ]
                )
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    from .helpers.runner import run

    parser = argparse.ArgumentParser(
        description="Run MyAgent in a small ANAC OneShot test tournament."
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Simulation steps per world (default: runner default). "
        "Keep >20 and <200 so contexts stay valid.",
    )
    parser.add_argument(
        "--n-configs",
        type=int,
        default=None,
        help="Number of world configurations to try (default: runner default).",
    )
    args = parser.parse_args()

    # Only forward args the user actually supplied, so with no CLI args the
    # runner falls back entirely to its own defaults.
    kwargs = {}
    if args.n_steps is not None:
        kwargs["n_steps"] = args.n_steps
    if args.n_configs is not None:
        kwargs["n_configs"] = args.n_configs

    run([MyAgent], **kwargs)
