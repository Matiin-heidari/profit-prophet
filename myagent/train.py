# trains an RL model
#
import logging
import os
from typing import Any

from negmas.sao import SAOResponse
from rich import print
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.rl.action import FlexibleActionManager
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.reward import DefaultRewardFunction
from scml.oneshot.context import GeneralContext, StrongSupplierContext, BalancedSupplierContext, WeakSupplierContext, StrongConsumerContext, BalancedConsumerContext, WeakConsumerContext

from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from multiprocessing import Process, Queue

# sys.path.append(str(Path(__file__).parent))
from .common import MODEL_PATH, CONTEXTS, MyObservationManager, TrainingAlgorithm, get_parallelization_params, make_context

NTRAINING = 100  # number of training steps


class ProgressCallback(BaseCallback):
    def __init__(self, queue: Queue, context_name: str):
        super().__init__()
        self.queue = queue
        self.context_name = context_name

    def _on_step(self) -> bool:
        self.queue.put((self.context_name, self.training_env.num_envs))
        return True

    def _on_training_end(self):
        pass


class MyRewardFunction(DefaultRewardFunction):
    """Reward shaping using score improvement."""

    def __init__(self, context: GeneralContext):
        super().__init__()
        self.context = context

    def before_action(self, awi: OneShotAWI) -> float:
        return float(getattr(awi, "current_score", 0.0))

    def __call__(self, awi: OneShotAWI, action: dict[str, SAOResponse], info: float):
        base_reward = super().__call__(awi, action, info)

        previous_score = float(info or 0.0)
        current_score = float(getattr(awi, "current_score", previous_score))
        score_delta = current_score - previous_score

        return base_reward + 0.1 * score_delta


def make_env(context_name, log: bool = False) -> OneShotEnv:
    log_params: dict[str, Any] = (
        dict(
            no_logs=False,
            log_stats_every=1,
            log_file_level=logging.DEBUG,
            log_screen_level=logging.ERROR,
            save_signed_contracts=True,
            save_cancelled_contracts=True,
            save_negotiations=True,
            save_resolved_breaches=True,
            save_unresolved_breaches=True,
            debug=True,
        )
        if log
        else dict(debug=True)
    )
    log_params.update(
        dict(
            ignore_agent_exceptions=False,
            ignore_negotiation_exceptions=False,
            ignore_contract_execution_exceptions=False,
            ignore_simulation_exceptions=False,
        )
    )
    context = make_context(context_name)
    return OneShotEnv(
        action_manager=FlexibleActionManager(context=context),
<<<<<<< HEAD
        observation_manager=MyObservationManager(context=context),  # type: ignore
        reward_function=MyRewardFunction(context=context),
=======
        observation_manager=MyObservationManager(context=context, continuous=True),  # type: ignore
        reward_function=MyRewardFunction(),
>>>>>>> observation-manager-improvements
        context=context,
        extra_checks=False,
    )


def try_a_model(
    model,
    context_name: str,
):
    """Runs a single simulation with one agent controlled with the given model"""

    obs_type = MyObservationManager
    # Create a world context compatibly with the model
    context = make_context(context_name)
    # sample a world and the RL agents (always one in this case)
    world, _ = context.generate(
        types=(OneShotRLAgent,),
        params=(
            dict(
                models=[model_wrapper(model)],
                observation_managers=[obs_type(context, continuous=True)],
                action_managers=[FlexibleActionManager(context)],
            ),
        ),
    )
    # run the world simulation
    world.run_with_progress()
    return world

def train_one(context_name, ntrain, params, queue):
    print(f"Training as {context_name}")
    env = None

    try:
        env = SubprocVecEnv(
            [lambda: make_env(context_name)] * params["n_envs"]
        )

        model = TrainingAlgorithm(
            "MlpPolicy", env, verbose=0
        )

        model.learn(
            total_timesteps=ntrain,
            progress_bar=False,
            callback=ProgressCallback(queue, context_name),
        )

        model_path = MODEL_PATH.parent / f"{MODEL_PATH.name}{context_name}"
        model.save(model_path)

    finally:
        if env is not None:
            env.close()

        queue.put((context_name, None))


def main(ntrain: int = NTRAINING):
    # choose the type of the model. Possibilities supported are:
    # fixed: Supports a single world configuration
    # limited: Supports a limited range of world configuration
    # unlimited: Supports any range of world configurations

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    
    if slurm_cpus:
        total_cores = int(slurm_cpus)
    else:
        total_cores = os.cpu_count() or 1
    n_parallel = min(len(CONTEXTS), max(1, (total_cores - 2) // 2), 3)
    params = get_parallelization_params(n_models_parallel=n_parallel)

    queue = Queue()

    for i in range(0, len(CONTEXTS), n_parallel):
        batch = CONTEXTS[i : i + n_parallel]

        # create one bar per context in this batch
        bars = {
            name: tqdm(total=ntrain, desc=name, position=j, leave=True)
            for j, name in enumerate(batch)
        }

        processes = [
            Process(target=train_one, args=(context_name, ntrain, params, queue))
            for context_name in batch
        ]
        for p in processes:
            p.start()

        # main process handles all terminal output
        finished = 0
        while finished < len(batch):
            context_name, steps = queue.get()
            if steps is None:
                bars[context_name].close()
                finished += 1
            else:
                bars[context_name].update(steps)

        for p in processes:
            p.join()


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else NTRAINING)
