"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML competition.
"""

from pathlib import Path

from scml.oneshot.rl.action import FlexibleActionManager
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper
from scml.oneshot.rl.observation import FlexibleObservationManager

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


if __name__ == "__main__":
    from .helpers.runner import run

    run([MyAgent], n_steps=30, n_configs=10)
