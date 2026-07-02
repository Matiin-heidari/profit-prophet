"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML competition.
"""

import csv
import os
from pathlib import Path

from scml.oneshot.rl.action import FlexibleActionManager
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper

from .common import (
    ALL_CONTEXTS,
    LOG_ROOT,
    MODEL_PATH,
    MyObservationManager,
    TrainingAlgorithm,
    make_context,
)


class MyAgent(OneShotRLAgent):
    """RL agent that loads one trained model per context."""

    def __init__(self, *args, **kwargs):
        base_name = MODEL_PATH.name
        self.paths: list[Path] = []

        observation_managers = []
        action_managers = []

        # The runtime agent should always load all supported contexts.
        for context_name in ALL_CONTEXTS:
            model_path = MODEL_PATH.parent / f"{base_name}{context_name}"
            self.paths.append(model_path)

            context = make_context(context_name)
            observation_managers.append(MyObservationManager(context, continuous=True))
            action_managers.append(FlexibleActionManager(context))

        models = tuple(
            model_wrapper(TrainingAlgorithm.load(path))
            for path in self.paths
        )

        kwargs.update(
            dict(
                models=models,
                observation_managers=observation_managers,
                action_managers=action_managers,
            )
        )

        super().__init__(*args, **kwargs)

    def init(self):
        super().init()
        self._log_context_usage()

    def context_switch(self) -> None:
        """Select the per-context model from THIS agent's actual position.

        We override the base `OneShotRLAgent.context_switch`, which routes via
        `context.is_valid_awi(awi, types=(type(self),))`. That path is broken for
        deployment: worlds store agents wrapped in `DefaultOneShotAdapter`, so the
        `find_test_agents` type-match (`isinobject(adapter, MyAgent)`) never hits,
        the per-agent validity loop is skipped, and EVERY context validates
        vacuously — so the base always picks index 0 (StrongSupplier) for every
        world, including consumer positions. See CLAUDE.md.

        Instead we route directly off `self.awi`:
          * side  = supplier (first level) vs consumer (last level), from `level`;
          * strength = Strong/Balanced/Weak from the trading-side partner count
            relative to same-level competitors (`d = n_side - n_competitors`),
            with thresholds derived empirically from generated worlds
            (d<=-1 Strong, d in {0,1} Balanced, d>=2 Weak). Side is exact;
            strength has irreducible overlap between the contexts near the
            boundaries, so it is best-effort but always lands on the right side.
        """
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
            run = os.environ.get("RUN_NAME", "default")
            log_dir = Path(LOG_ROOT) / "context_usage_logs" / run
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / f"usage_{os.getpid()}.csv"
            is_new = not path.exists()
            with open(path, "a", newline="") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(
                        ["agent_id", "chosen", "level", "n_suppliers", "n_consumers"]
                    )
                writer.writerow(
                    [
                        self.id,
                        chosen,
                        getattr(awi, "level", ""),
                        len(getattr(awi, "my_suppliers", []) or []),
                        len(getattr(awi, "my_consumers", []) or []),
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
