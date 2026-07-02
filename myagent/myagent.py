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
            log_dir = Path("context_usage_logs") / run
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
    from .helpers.runner import run

    run([MyAgent])
