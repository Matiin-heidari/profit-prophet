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

NTRAINING = 300000  # number of training steps


class ProgressCallback(BaseCallback):
    def __init__(self, queue: Queue, context_name: str):
        super().__init__()
        self.queue = queue
        self.context_name = context_name

    def _on_step(self) -> bool:
        self.queue.put((self.context_name, self.training_env.num_envs))
        return True

    def _on_training_end(self):
        self.queue.put((self.context_name, None))  # signal done


class MyRewardFunction(DefaultRewardFunction):
    """Reward shaping on top of SCML's default reward."""

    def __init__(self, context: GeneralContext):
        super().__init__()
        self.context = context

    def before_action(self, awi: OneShotAWI) -> float:
        return super().before_action(awi)

    def __call__(self, awi: OneShotAWI, action: dict[str, SAOResponse], info: float):
        base_reward = super().__call__(awi, action, info)

        needed_sales = max(0, getattr(awi, "needed_sales", 0))
        needed_supplies = max(0, getattr(awi, "needed_supplies", 0))
        time_pressure = float(getattr(awi, "relative_time", 0.0))

        if isinstance(self.context, (StrongSupplierContext, StrongConsumerContext)):
            quantity_weight = 0.03
        elif isinstance(self.context, (BalancedSupplierContext, BalancedConsumerContext)):
            quantity_weight = 0.05
        elif isinstance(self.context, (WeakSupplierContext, WeakConsumerContext)):
            quantity_weight = 0.08
        else:
            quantity_weight = 0.05

        shaping = 0.0

        if isinstance(self.context, (StrongSupplierContext, BalancedSupplierContext, WeakSupplierContext)):
            shaping -= quantity_weight * needed_sales * (0.5 + time_pressure)

        elif isinstance(self.context, (StrongConsumerContext, BalancedConsumerContext, WeakConsumerContext)):
            shaping -= quantity_weight * needed_supplies * (0.5 + time_pressure)

        imbalance = abs(needed_sales - needed_supplies)
        shaping -= 0.01 * imbalance

        return base_reward + shaping

        
        
        
        

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
        observation_manager=MyObservationManager(context=context),  # type: ignore
        reward_function=MyRewardFunction(context=context),
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
                observation_managers=[obs_type(context)],
                action_managers=[FlexibleActionManager(context)],
            ),
        ),
    )
    # run the world simulation
    world.run_with_progress()
    return world

def train_one(context_name, ntrain, params, queue):
        print(f"Training as {context_name}")
        # create a gymnasium environment for training
        env = env = SubprocVecEnv(
            [lambda: make_env(context_name)] * params["n_envs"]
        )

        # choose a training algorithm
        model = TrainingAlgorithm(  # type: ignore learning_rate must be passed by the algorithm itself
            "MlpPolicy", env, verbose=0
        )

        # train the model
        model.learn(
            total_timesteps=ntrain,
            progress_bar=False,
            callback=ProgressCallback(queue, context_name),
        )
        #print(f"\tFinished training the model for {ntrain} steps ... Testing it on a single world simulation")

        # decide the model path to save to
        model_path = (
            MODEL_PATH.parent
            / f"{MODEL_PATH.name}{context_name}"
        )

        # save the model
        model.save(model_path)
        #model = TrainingAlgorithm.load(model_path)
        # try the model in a single simulation
        #world = try_a_model(model, context_name)
        #print(world.scores())


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
