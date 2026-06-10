# trains an RL model
#
import logging
import os, json
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
from scml.oneshot.rl.observation import FlexibleObservationManager

from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from multiprocessing import Process, Queue
import numpy as np

# sys.path.append(str(Path(__file__).parent))
from .common import MODEL_PATH, CONTEXTS, TrainingAlgorithm, get_parallelization_params, make_context

NTRAINING = 300000  # number of training steps


_UNIT_PRICE_IDX = 2

def _catalog_prices(awi: OneShotAWI) -> tuple[float, float]:
    """Return (input_catalog_price, output_catalog_price) for this agent.

    Falls back to (1.0, 1.0) if the AWI doesn't expose the attribute.
    """
    try:
        prices = awi.catalog_prices
        level = int(awi.level)
        return float(prices[level]), float(prices[level + 1])
    except Exception:
        return 1.0, 1.0
    
def _trading_prices(awi: OneShotAWI) -> tuple[float, float]:
    try:
        prices = awi.trading_prices
        return float(prices[awi.my_input_product]), float(prices[awi.my_output_product])
    except Exception:
        return _catalog_prices(awi)
    
def _sell_offer_prices(awi: OneShotAWI) -> list[float]:
    """Unit prices from all non-None current sell offers."""
    try:
        offers = awi.current_sell_offers or {}
        return [
            float(o[_UNIT_PRICE_IDX])
            for o in offers.values()
            if o is not None
        ]
    except Exception:
        return []
 
 
def _buy_offer_prices(awi: OneShotAWI) -> list[float]:
    """Unit prices from all non-None current buy offers."""
    try:
        offers = awi.current_buy_offers or {}
        return [
            float(o[_UNIT_PRICE_IDX])
            for o in offers.values()
            if o is not None
        ]
    except Exception:
        return []
 
def _needed_sales(awi: OneShotAWI) -> float:
    """Quantity the agent still needs to sell this step (0 if fulfilled)."""
    try:
        return max(0.0, float(getattr(awi, "needed_sales", 0) or 0))
    except Exception:
        return 0.0
 
 
def _needed_supplies(awi: OneShotAWI) -> float:
    """Quantity the agent still needs to buy this step (0 if fulfilled)."""
    try:
        return max(0.0, float(getattr(awi, "needed_supplies", 0) or 0))
    except Exception:
        return 0.0
 
def _shortfall_sell_ratio(awi: OneShotAWI) -> float:
    """Fraction of required sales not yet covered, in [0, 1]."""
    try:
        required = float(getattr(awi, "current_exogenous_input_quantity", 0) or 0)
        needed = float(getattr(awi, "needed_sales", 0) or 0)
        return float(np.clip(needed / max(required, 1.0), 0.0, 1.0))
    except Exception:
        return 0.0


def _shortfall_buy_ratio(awi: OneShotAWI) -> float:
    """Fraction of required supplies not yet secured, in [0, 1]."""
    try:
        required = float(getattr(awi, "current_exogenous_output_quantity", 0) or 0)
        needed = float(getattr(awi, "needed_supplies", 0) or 0)
        return float(np.clip(needed / max(required, 1.0), 0.0, 1.0))
    except Exception:
        return 0.0



class _BaseReward(DefaultRewardFunction):
    """Base reward functions inherited by all context specific reward functions.

    ``before_action`` saves the current score
     
    ``__call__`` computes: reward = "BASE_WEIGHT * default_reward + DELTA_WEIGHT * score_delta + _extra(...)"
    """

   
    BASE_WEIGHT: float = 1.0
    DELTA_WEIGHT: float = 0.1  # Override in subclasses to tune urgency / patience. Low = Patient; High = Urgent

    def __init__(self, context: GeneralContext) -> None:
        super().__init__()
        self.context = context


    def before_action(self, awi: OneShotAWI) -> float:  
        return float(getattr(awi, "current_score", 0.0))

    def __call__(
        self,
        awi: OneShotAWI,
        action: dict[str, SAOResponse],
        info: float,
    ) -> float:
        base = super().__call__(awi, action, info)
        prev_score = float(info or 0.0)
        curr_score = float(getattr(awi, "current_score", prev_score))
        delta = curr_score - prev_score
        extra = self._extra(awi, action)
        return self.BASE_WEIGHT * base + self.DELTA_WEIGHT * delta + extra

    def _extra(self, awi: OneShotAWI, action: dict[str, SAOResponse]) -> float:
        """Context-specific shaping term.  Must not raise."""
        return 0.0


class StrongSupplierRewardFunction(_BaseReward):
    """Reward for a *strong* supplier position. Meaning the consumers need our products more than we need them.
    We wait for a good price, reward prices above and penalise prices below the catalogue price.

    """
 
    DELTA_WEIGHT = 0.05
    PRICE_SCALE = 0.20    # max bonus/penalty magnitude per agreement
 
    def _extra(self, awi: OneShotAWI, action: dict[str, SAOResponse]) -> float:
        try:
            _, catalog_out = _trading_prices(awi)
            bonus = 0.0
            # This rewards being in negotiations with above catalogue prices instead of having completed good deals
            for price in _sell_offer_prices(awi):
                ratio = (price - catalog_out) / max(catalog_out, 1e-6)
                bonus += float(np.clip(ratio, -0.10, 0.10)) * self.PRICE_SCALE
            return bonus
        except Exception:
            return 0.0


 

class WeakSupplierRewardFunction(_BaseReward):
    """Reward for a *weak* supplier position. Demand for the product is low and
    unsold products incure a penalty. So we prioritise getting any agreement at all instead of
    good pricing. We penalise any unsold products.
    """

    DELTA_WEIGHT = 0.20       
    SHORTFALL_SCALE = 0.20 # scale of penalty for unsold products

    def _extra(self, awi: OneShotAWI, action: dict[str, SAOResponse]) -> float:
        try:
            return -self.SHORTFALL_SCALE * _shortfall_sell_ratio(awi)
        except Exception:
            return 0.0



class BalancedSupplierRewardFunction(_BaseReward):
    """Reward for a *balanced* supplier position. Combines a moderate reward for above-catalog sells
    with a moderate shortfall penalty.

    Small price deviation bonus (half the StrongSupplier scale) plus a
    small flat per-agreement bonus to avoid zero-volume solutions."""


    DELTA_WEIGHT = 0.10
    PRICE_SCALE = 0.10
    SHORTFALL_SCALE = 0.10
 
    def _extra(self, awi: OneShotAWI, action: dict[str, SAOResponse]) -> float:
        try:
            _, catalog_out = _trading_prices(awi)
 
            prices = _sell_offer_prices(awi)
            price_bonus = 0.0
            if prices:
                mean_ratio = (np.mean(prices) - catalog_out) / max(catalog_out, 1e-6)
                price_bonus = float(np.clip(mean_ratio, -0.05, 0.05)) * self.PRICE_SCALE
 
            shortfall_penalty = -self.SHORTFALL_SCALE * _shortfall_sell_ratio(awi)
 
            return price_bonus + shortfall_penalty
        except Exception:
            return 0.0

class StrongConsumerRewardFunction(_BaseReward):
    """Reward for a *strong* consumer position. More producers than consumers. We aim for below-cataloge prices.
     """
 
    DELTA_WEIGHT = 0.05
    PRICE_SCALE = 0.20
 
    def _extra(self, awi: OneShotAWI, action: dict[str, SAOResponse]) -> float:
        try:
            catalog_in, _ = _trading_prices(awi)
            bonus = 0.0
            for price in _buy_offer_prices(awi):
                ratio = (catalog_in - price) / max(catalog_in, 1e-6)
                bonus += float(np.clip(ratio, -0.10, 0.10)) * self.PRICE_SCALE
            return bonus
        except Exception:
            return 0.0


class WeakConsumerRewardFunction(_BaseReward):
    """Reward shaping for a *weak* consumer position. Input supply is scarce. Failure to secure enough
    inputs triggers shortfall penalties and prevents fulfilment of output contracts.
    We heavily penalise a large ``needed_supplies`` value.
    """

    DELTA_WEIGHT = 0.20
    SHORTFALL_SCALE = 0.20

    def _extra(self, awi: OneShotAWI, action: dict[str, SAOResponse]) -> float:
        try:
            shortfall_ratio = _shortfall_buy_ratio(awi)
            return -self.SHORTFALL_SCALE * shortfall_ratio
        except Exception:
            return 0.0

class BalancedConsumerRewardFunction(_BaseReward):
    """Reward for a *balanced* supplier position. Combines a moderate reward for below-catalog buys
    with a moderate shortfall penalty.
    """

    DELTA_WEIGHT = 0.10
    PRICE_SCALE = 0.10
    SHORTFALL_SCALE = 0.10

    def _extra(self, awi: OneShotAWI, action: dict[str, SAOResponse]) -> float:
        try:
            catalog_in, _ = _trading_prices(awi)

            prices = _buy_offer_prices(awi)
            price_bonus = 0.0
            if prices:
                mean_ratio = (catalog_in - np.mean(prices)) / max(catalog_in, 1e-6)
                price_bonus = float(np.clip(mean_ratio, -0.05, 0.05)) * self.PRICE_SCALE

            shortfall_penalty = -self.SHORTFALL_SCALE * _shortfall_buy_ratio(awi)

            return price_bonus + shortfall_penalty
        except Exception:
            return 0.0

_CONTEXT_TO_REWARD: dict[type, type[_BaseReward]] = {
    StrongSupplierContext: StrongSupplierRewardFunction,
    BalancedSupplierContext: BalancedSupplierRewardFunction,
    WeakSupplierContext: WeakSupplierRewardFunction,
    StrongConsumerContext: StrongConsumerRewardFunction,
    BalancedConsumerContext: BalancedConsumerRewardFunction,
    WeakConsumerContext: WeakConsumerRewardFunction,
}

def make_reward_function(context: GeneralContext) -> _BaseReward:
    """Return the reward function best suited to *context*.

    Falls back to :class:`_BaseReward` (score + delta only) for any context
    type not in the registry.
    """
    reward_cls = _CONTEXT_TO_REWARD.get(type(context), _BaseReward)
    return reward_cls(context)

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


class EvaluationCallback(BaseCallback):
    def __init__(self, context_name: str, eval_freq: int = 10_000, n_eval_episodes: int = 10):
        super().__init__()
        self.context_name = context_name
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    AGENT_KEYS = (
        "score", "score_vs_mean_opponent", "score_rank", "bankrupt",
        "shortfall_penalty", "shortfall_quantity", "disposal_cost", "productivity",
        "neg_requests_received", "neg_requests_rejected", "neg_requests_sent",
        "negs_initiated", "negs_failed", "agent_agreement_rate",
    )
    WORLD_KEYS = (
        "welfare", "relative_welfare",
        "n_negotiation_successful", "n_negotiation_failed",
        "n_negotiation_rounds_successful", "n_negotiation_rounds_failed",
        "n_contracts_nullified", "activity_level",
    )

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            results = [evaluate_model(self.model, self.context_name)
                       for _ in range(self.n_eval_episodes)]

            for key in self.AGENT_KEYS:
                vals = [r[key] for r in results if r.get(key) is not None]
                if vals:
                    self.logger.record(f"agent/{key}", float(np.mean(vals)))

            for key in self.WORLD_KEYS:
                vals = [r[key] for r in results if r.get(key) is not None]
                if vals:
                    self.logger.record(f"world/{key}", float(np.mean(vals)))

            self.logger.dump(self.num_timesteps)

        return True

""""""
class MyRewardFunction(DefaultRewardFunction):
    """Reward shaping using score improvement."""

    def __init__(self, context: GeneralContext):
        super().__init__()
        self.context = context

    def before_action(self, awi: OneShotAWI) -> float:
        return float(getattr(awi, "current_score", 0.0))

    def __call__(self, awi: OneShotAWI, action: dict[str, SAOResponse], info: float):
        base_reward = super().__call__(awi, action, info)
        """
        snapshot = dump_object(awi)
        if awi.current_offers != {}: print(f"Offers: {awi.current_offers}")
        print(f"Lines: {awi.n_lines}")
        print(f"Level:{awi.level} Total Sales:{awi.total_sales} ExInput: {awi.current_exogenous_input_quantity} Needed Sales: {awi.needed_sales}")

        with open("awi_dump.json", "w") as f:
            json.dump(snapshot, f, indent=4, default=str)"""

        previous_score = float(info or 0.0)
        current_score = float(getattr(awi, "current_score", previous_score))
        score_delta = current_score - previous_score

        return base_reward + 0.1 * score_delta

def dump_object(obj):
    data = {}

    for attr in dir(obj):
        if attr.startswith("_"):
            continue

        try:
            value = getattr(obj, attr)
            if callable(value):
                continue
            data[attr] = value
        except Exception as e:
            data[attr] = f"<error: {e}>"

    return data

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
        observation_manager=FlexibleObservationManager(context=context), 
        reward_function=make_reward_function(context=context),
        context=context,
        extra_checks=False,
    )


def evaluate_model(model, context_name: str) -> dict:
    """Runs a single simulation with one agent controlled with the given model. Similar to try_a_model but without progress output."""
    context = make_context(context_name)
    world, agents = context.generate(
        types=(OneShotRLAgent,),
        params=(
            dict(
                models=[model_wrapper(model)],
                observation_managers=[FlexibleObservationManager(context)],
                action_managers=[FlexibleActionManager(context)],
            ),
        ),
    )

    world.run_with_progress()

    agent_id = agents[0].id

    result = {}

    all_scores = world.scores()
    opponent_scores = [v for k, v in all_scores.items() if k != agent_id]

    result["score"] = all_scores[agent_id]
    result["score_vs_mean_opponent"] = all_scores[agent_id] - float(np.mean(opponent_scores)) if opponent_scores else float("nan")
    result["score_rank"] = sorted(all_scores.values(), reverse=True).index(all_scores[agent_id]) + 1
    result["bankrupt"] = world.is_bankrupt[agent_id]
    result["shortfall_penalty"] = sum(world.stats[f"shortfall_penalty_{agent_id}"])
    result["shortfall_quantity"] = sum(world.stats[f"shortfall_quantity_{agent_id}"])
    result["disposal_cost"] = sum(world.stats[f"disposal_cost_{agent_id}"])
    result["productivity"] = float(np.mean(world.stats[f"productivity_{agent_id}"]))
    result["n_negotiation_successful"] = sum(world.stats["n_negotiation_successful"])
    result["n_negotiation_failed"] = sum(world.stats["n_negotiation_failed"])
    result["n_negotiation_rounds_successful"] = sum(world.stats["n_negotiation_rounds_successful"])
    result["n_negotiation_rounds_failed"] = sum(world.stats["n_negotiation_rounds_failed"])
    result["n_contracts_nullified"] = sum(world.stats["n_contracts_nullified_now"])
    result["activity_level"] = float(np.mean(world.stats["activity_level"]))
    result["neg_requests_received"] = world.neg_requests_received[agent_id]
    result["neg_requests_rejected"] = world.neg_requests_rejected[agent_id]
    result["neg_requests_sent"] = world.neg_requests_sent[agent_id]
    result["negs_initiated"] = world.negs_initiated[agent_id]
    result["negs_failed"] = world.negs_failed[agent_id]
    received = world.neg_requests_received[agent_id]
    failed = world.negs_failed[agent_id]
    result["agent_agreement_rate"] = (received - failed) / received if received > 0 else float("nan")
    result["welfare"] = world.welfare()
    result["relative_welfare"] = world.relative_welfare()

    return result


def try_a_model(
    model,
    context_name: str,
):
    """Runs a single simulation with one agent controlled with the given model"""

    obs_type = FlexibleObservationManager
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

def try_a_trained_model(context_name: str):
    """Runs a simulation with one agent controlled by the already trained model for the given context"""
    path = MODEL_PATH.parent / f"{MODEL_PATH.name}{BalancedSupplierContext}"
    model = TrainingAlgorithm.load(path)
    obs_type = FlexibleObservationManager
    # Create a world context compatibly with the model
    context = make_context(context_name)
    # sample a world and the RL agents (always one in this case)
    world, agent = context.generate(
        types=(OneShotRLAgent,),
        params=(
            dict(
                models=[model_wrapper(model)],
                observation_managers=[obs_type(context)],
                action_managers=[FlexibleActionManager(context)],
            ),
        ),
    )

    print(f"Assigned Agent: {agent[0].name}, {agent[0].id}")
    world_agents = world.agents
    print(world_agents)
    world.run_with_progress()
    
    scores = world.scores()
    print(scores)
    rl_score = world.scores()[agent[0].name]
    print(f"Our Score: {rl_score}")
    print(f"Agreement: {world.agreement_rate}")
    print(f"Bankrupt: {world.is_bankrupt[agent[0].name]}")
    print(f"neg_requests_received: {world.neg_requests_received[agent[0].name]}")
    print(f"neg_requests_rejected: {world.neg_requests_rejected[agent[0].name]}")
    print(f"neg_requests_sent: {world.neg_requests_sent[agent[0].name]}")
    print(f"negs_initiated: {world.negs_initiated[agent[0].name]}")
    print(f"negs_failed: {world.negs_failed[agent[0].name]}")


    return world

    
def train_one(context_name, ntrain, params, queue):
    print(f"Training as {context_name}")
    env = None

    try:
        env = SubprocVecEnv(
            [lambda: make_env(context_name)] * params["n_envs"]
        )      

        model = TrainingAlgorithm(  # type: ignore learning_rate must be passed by the algorithm itself
            "MlpPolicy", env, verbose=0, tensorboard_log=f"./tensorboard_logs/{context_name}"
        )

        model.learn(
            total_timesteps=ntrain,
            progress_bar=False,
            callback=[ProgressCallback(queue, context_name),
                      EvaluationCallback(context_name, eval_freq=int(NTRAINING/10), n_eval_episodes=40)
                      ] 
        )

        model_path = MODEL_PATH.parent / f"{MODEL_PATH.name}{context_name}"
        model.save(model_path)

    finally:
        if env is not None:
            env.close()

        queue.put((context_name, None))

def test_train(context_name):
    env = make_env(context_name)
    obs, _ = env.reset()
    world = env._world
    print(type(world))
    for agent_id, agent in world.agents.items():
        print(agent_id, type(agent).__name__)
    

def main(ntrain: int = NTRAINING):
    # choose the type of the model. Possibilities supported are:
    # fixed: Supports a single world configuration
    # limited: Supports a limited range of world configuration
    # unlimited: Supports any range of world configurations

    #test_train("StrongSupplierContext")

    
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    
    if slurm_cpus:
        total_cores = int(slurm_cpus)
    else:
        total_cores = os.cpu_count() or 1
    n_parallel = min(len(CONTEXTS), max(1, (total_cores - 2) // 2), 3)
    params = get_parallelization_params(n_models_parallel=n_parallel)
    
    #params = {"n_envs": 1,"n_models_parallel": 1,}
    #n_parallel = 1

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
    #cont_str = "BalancedSupplierContext"
    #try_a_trained_model(cont_str)
    #path = MODEL_PATH.parent / f"{MODEL_PATH.name}{cont_str}"
    #model = TrainingAlgorithm.load(path)
    #print(evaluate_model(model, cont_str))

    main(int(sys.argv[1]) if len(sys.argv) > 1 else NTRAINING)
