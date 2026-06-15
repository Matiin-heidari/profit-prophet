# trains an RL model
import os
from multiprocessing import Process, Queue
from typing import Any

import numpy as np
from negmas.sao import SAOResponse
from rich import print
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.context import GeneralContext
from scml.oneshot.rl.action import FlexibleActionManager
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.reward import DefaultRewardFunction
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from tqdm import tqdm

from .common import (
    MODEL_PATH,
    CONTEXTS,
    MyObservationManager,
    TrainingAlgorithm,
    get_parallelization_params,
    make_context,
)

NTRAINING = 300000  # number of training steps


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to finite float."""
    try:
        if value is None:
            return default

        value_float = float(value)

        if not np.isfinite(value_float):
            return default

        return value_float

    except (TypeError, ValueError):
        return default


def _numeric_values(value: Any) -> list[float]:
    """Convert scalar/list/array values to finite floats."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        raw_values = list(value)
    elif hasattr(value, "tolist"):
        raw_values = value.tolist()
    elif hasattr(value, "to_list"):
        raw_values = value.to_list()
    elif hasattr(value, "values"):
        try:
            raw_values = list(value.values)
        except Exception:
            raw_values = [value]
    else:
        raw_values = [value]

    values = []

    for raw_value in raw_values:
        try:
            value_float = float(raw_value)

            if np.isfinite(value_float):
                values.append(value_float)

        except (TypeError, ValueError):
            continue

    return values


def _mean_numeric(values: list[Any], default: float = 0.0) -> float:
    """Average numeric values only."""
    numeric_values = _numeric_values(values)

    if not numeric_values:
        return default

    return float(np.mean(numeric_values))


def _safe_numeric_summary(value: Any, default: float = 0.0) -> float:
    """Summarize scalar/list/array values as a mean."""
    values = _numeric_values(value)

    if not values:
        return default

    return float(np.mean(values))


def _extract_world_stat(world: Any, key: str) -> float | None:
    """Read optional world statistics."""
    for attr_name in ("stats", "statistics"):
        stats = getattr(world, attr_name, None)

        if isinstance(stats, dict) and key in stats:
            return _safe_numeric_summary(stats[key])

    return None


def _metric_safe_name(name: str) -> str:
    """Make names safe for TensorBoard tags."""
    safe_chars = []

    for char in name:
        if char.isalnum() or char in ("_", "-"):
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    return "".join(safe_chars) or "unknown"


def _agent_code(agent_id: str) -> str:
    """Extract the short SCML agent code from an agent id."""
    base = agent_id.split("@", 1)[0]
    return base.lstrip("0123456789")


def _rl_agent_code() -> str:
    """Return the score-code used for the RL agent."""
    return os.environ.get("RL_AGENT_CODE", "On")


def _is_rl_agent_score_key(agent_id: str) -> bool:
    """Detect the RL agent in world.scores()."""
    return _agent_code(agent_id) == _rl_agent_code()


def _rank_of_agents(scores: dict[str, float], agent_ids: list[str]) -> float:
    """Return the best 1-based rank of selected agents."""
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ranks = {agent_id: rank for rank, (agent_id, _) in enumerate(ranked, start=1)}

    selected_ranks = [
        ranks[agent_id]
        for agent_id in agent_ids
        if agent_id in ranks
    ]

    if not selected_ranks:
        return float(len(scores))

    return float(min(selected_ranks))


def _add_series_metrics(
    metrics: dict[str, float],
    name: str,
    value: Any,
) -> None:
    """Add summary metrics for a numeric series."""
    values = _numeric_values(value)

    if not values:
        return

    metrics[f"{name}_mean"] = float(np.mean(values))
    metrics[f"{name}_last"] = float(values[-1])
    metrics[f"{name}_min"] = float(np.min(values))
    metrics[f"{name}_max"] = float(np.max(values))
    metrics[f"{name}_sum"] = float(np.sum(values))
    metrics[f"{name}_count"] = float(len(values))


def _add_my_agent_stat_metrics(
    metrics: dict[str, float],
    stats: dict[str, Any],
    agent_ids: list[str],
    key: str,
) -> None:
    """Aggregate a statistic over all RL-agent ids."""
    values = []

    for agent_id in agent_ids:
        stat_key = f"{key}_{agent_id}"

        if stat_key in stats:
            values.extend(_numeric_values(stats[stat_key]))

    if values:
        _add_series_metrics(metrics, f"my/{key}", values)


def evaluate_model(model, context_name: str) -> dict[str, float]:
    """Run one evaluation world and return loggable metrics."""
    context = make_context(context_name)

    world, _ = context.generate(
        types=(OneShotRLAgent,),
        params=(
            dict(
                models=[model_wrapper(model)],
                observation_managers=[MyObservationManager(context, continuous=True)],
                action_managers=[FlexibleActionManager(context)],
            ),
        ),
    )

    if hasattr(world, "run"):
        world.run()
    else:
        world.run_with_progress()

    metrics: dict[str, float] = {}
    scores: dict[str, float] = {}
    my_agent_ids: list[str] = []

    if hasattr(world, "scores"):
        raw_scores = world.scores()
        scores = {
            str(agent_id): float(score)
            for agent_id, score in raw_scores.items()
            if np.isfinite(float(score))
        }

        my_agent_ids = [
            agent_id
            for agent_id in scores
            if _is_rl_agent_score_key(agent_id)
        ]

        opponent_ids = [
            agent_id
            for agent_id in scores
            if agent_id not in my_agent_ids
        ]

        my_scores = [scores[agent_id] for agent_id in my_agent_ids]
        opponent_scores = [scores[agent_id] for agent_id in opponent_ids]

        metrics["n_agents"] = float(len(scores))
        metrics["n_rl_agents"] = float(len(my_agent_ids))
        metrics["world_score_mean"] = _mean_numeric(list(scores.values()))

        if scores:
            metrics["top_score"] = float(max(scores.values()))
            metrics["bottom_score"] = float(min(scores.values()))

        if my_scores:
            my_score = _mean_numeric(my_scores)
            metrics["score"] = my_score
            metrics["my_score"] = my_score
            metrics["my_rank"] = _rank_of_agents(scores, my_agent_ids)

        if opponent_scores:
            opponent_score = _mean_numeric(opponent_scores)
            metrics["opponent_score_mean"] = opponent_score
            metrics["best_opponent_score"] = float(max(opponent_scores))
            metrics["worst_opponent_score"] = float(min(opponent_scores))

            if my_scores:
                metrics["score_gap"] = metrics["my_score"] - opponent_score
                metrics["score_gap_vs_best_opponent"] = (
                    metrics["my_score"] - metrics["best_opponent_score"]
                )

        opponent_scores_by_code: dict[str, list[float]] = {}

        for opponent_id in opponent_ids:
            code = _metric_safe_name(_agent_code(opponent_id))
            opponent_scores_by_code.setdefault(code, []).append(scores[opponent_id])

        for code, typed_scores in opponent_scores_by_code.items():
            typed_score_mean = _mean_numeric(typed_scores)

            metrics[f"opponent/{code}_score_mean"] = typed_score_mean
            metrics[f"opponent/{code}_count"] = float(len(typed_scores))

            if my_scores:
                metrics[f"score_gap_vs/{code}"] = metrics["my_score"] - typed_score_mean

    stats = getattr(world, "stats", None)

    if isinstance(stats, dict):
        # World-level metrics.
        for key in (
            "n_negotiation_successful",
            "n_negotiation_failed",
            "n_negotiation_rounds_successful",
            "n_negotiation_rounds_failed",
            "n_contracts_signed",
            "n_contracts_concluded",
            "n_contracts_executed",
            "n_contracts_cancelled",
            "n_contracts_dropped",
            "n_contracts_nullified",
            "agreement_rate",
            "agreement_fraction",
            "contract_execution_fraction",
            "productivity",
            "welfare",
            "relative_welfare",
        ):
            if key in stats:
                _add_series_metrics(metrics, f"world/{key}", stats[key])

        # RL-agent-specific metrics.
        for key in (
            "score",
            "balance",
            "bankrupt",
            "productivity",
            "shortfall_quantity",
            "shortfall_penalty",
            "storage_cost",
            "disposal_cost",
            "inventory_penalized",
            "inventory_input",
            "inventory_output",
        ):
            _add_my_agent_stat_metrics(metrics, stats, my_agent_ids, key)

    return metrics


class ProgressCallback(BaseCallback):
    """Send training progress to the main process."""

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
    """Log evaluation metrics to TensorBoard."""

    def __init__(
        self,
        context_name: str,
        eval_freq: int,
        n_eval_episodes: int,
    ):
        super().__init__()
        self.context_name = context_name
        self.eval_freq = max(1, eval_freq)
        self.n_eval_episodes = max(1, n_eval_episodes)
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True

        self.last_eval_step = self.num_timesteps

        try:
            results = [
                evaluate_model(self.model, self.context_name)
                for _ in range(self.n_eval_episodes)
            ]

            metric_names = sorted({key for result in results for key in result})

            for metric_name in metric_names:
                values = [
                    float(result[metric_name])
                    for result in results
                    if metric_name in result and np.isfinite(float(result[metric_name]))
                ]

                if not values:
                    continue

                mean_value = float(np.mean(values))

                self.logger.record(f"eval/{metric_name}", mean_value)
                self.logger.record(f"eval/{metric_name}_mean", mean_value)
                self.logger.record(f"eval/{metric_name}_std", float(np.std(values)))
                self.logger.record(f"eval/{metric_name}_min", float(np.min(values)))
                self.logger.record(f"eval/{metric_name}_max", float(np.max(values)))

            self.logger.record("eval/failed", 0)

        except Exception as e:
            # Logging must not stop training.
            self.logger.record("eval/failed", 1)
            print(f"[eval failed] {self.context_name}: {e}")

        self.logger.dump(self.num_timesteps)
        return True


class TrainingDiagnosticsCallback(BaseCallback):
    """Log interval-based observation, action and reward diagnostics."""

    def __init__(self, log_freq: int):
        super().__init__()
        self.log_freq = max(1, log_freq)
        self.last_log_step = 0

        self.reward_buffer: list[float] = []
        self.action_buffer: list[float] = []
        self.done_buffer: list[float] = []

    def _on_step(self) -> bool:
        obs = self.locals.get("new_obs")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        actions = self.locals.get("actions")

        self._extend_buffer(self.reward_buffer, rewards)
        self._extend_buffer(self.action_buffer, actions)
        self._extend_buffer(self.done_buffer, dones)

        if self.num_timesteps - self.last_log_step < self.log_freq:
            return True

        self.last_log_step = self.num_timesteps

        # Snapshot diagnostics for the current observation.
        self._log_array_snapshot("diagnostics/obs", obs, log_features=True)

        # Interval diagnostics over all steps since the last log.
        self._log_interval("diagnostics/reward_interval", self.reward_buffer, log_signs=True)
        self._log_interval("diagnostics/action_interval", self.action_buffer)
        self._log_interval("diagnostics/done_interval", self.done_buffer)

        # Backward-compatible aliases for quick checks.
        self._log_interval("diagnostics/reward", self.reward_buffer, log_signs=True)

        if self.done_buffer:
            done_array = np.asarray(self.done_buffer, dtype=np.float32)
            self.logger.record("diagnostics/done_fraction", float(np.mean(done_array)))

        self.reward_buffer.clear()
        self.action_buffer.clear()
        self.done_buffer.clear()

        self.logger.dump(self.num_timesteps)
        return True

    def _extend_buffer(self, buffer: list[float], value: Any) -> None:
        """Append finite numeric values to a buffer."""
        if value is None:
            return

        try:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return

        finite_values = array[np.isfinite(array)]

        if finite_values.size == 0:
            return

        buffer.extend(float(value) for value in finite_values)

    def _log_interval(
        self,
        prefix: str,
        values: list[float],
        log_signs: bool = False,
    ) -> None:
        """Log summary statistics for an interval buffer."""
        self.logger.record(f"{prefix}_count", float(len(values)))

        if not values:
            return

        array = np.asarray(values, dtype=np.float32)
        finite_array = array[np.isfinite(array)]

        if finite_array.size == 0:
            return

        self.logger.record(f"{prefix}_mean", float(np.mean(finite_array)))
        self.logger.record(f"{prefix}_std", float(np.std(finite_array)))
        self.logger.record(f"{prefix}_min", float(np.min(finite_array)))
        self.logger.record(f"{prefix}_max", float(np.max(finite_array)))
        self.logger.record(f"{prefix}_sum", float(np.sum(finite_array)))
        self.logger.record(f"{prefix}_abs_mean", float(np.mean(np.abs(finite_array))))

        if log_signs:
            eps = 1e-12

            self.logger.record(
                f"{prefix}_nonzero_fraction",
                float(np.mean(np.abs(finite_array) > eps)),
            )
            self.logger.record(
                f"{prefix}_zero_fraction",
                float(np.mean(np.abs(finite_array) <= eps)),
            )
            self.logger.record(
                f"{prefix}_positive_fraction",
                float(np.mean(finite_array > eps)),
            )
            self.logger.record(
                f"{prefix}_negative_fraction",
                float(np.mean(finite_array < -eps)),
            )

    def _log_array_snapshot(
        self,
        prefix: str,
        value: Any,
        log_features: bool = False,
    ) -> None:
        """Log summary stats for the current array snapshot."""
        if value is None:
            return

        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            return

        if array.size == 0:
            return

        finite_mask = np.isfinite(array)
        finite_values = array[finite_mask]

        self.logger.record(f"{prefix}_nan_count", float(np.isnan(array).sum()))
        self.logger.record(f"{prefix}_inf_count", float(np.isinf(array).sum()))

        if finite_values.size == 0:
            return

        self.logger.record(f"{prefix}_mean", float(np.mean(finite_values)))
        self.logger.record(f"{prefix}_std", float(np.std(finite_values)))
        self.logger.record(f"{prefix}_min", float(np.min(finite_values)))
        self.logger.record(f"{prefix}_max", float(np.max(finite_values)))

        if prefix == "diagnostics/obs":
            self.logger.record(
                "diagnostics/obs_low_clip_fraction",
                float(np.mean(finite_values <= 0.0)),
            )
            self.logger.record(
                "diagnostics/obs_high_clip_fraction",
                float(np.mean(finite_values >= 1.0)),
            )

        if not log_features:
            return

        if array.ndim != 2:
            return

        feature_means = np.nanmean(array, axis=0)
        feature_stds = np.nanstd(array, axis=0)
        feature_mins = np.nanmin(array, axis=0)
        feature_maxs = np.nanmax(array, axis=0)

        for idx, value_mean in enumerate(feature_means):
            self.logger.record(
                f"{prefix}_feature_{idx:02d}_mean",
                _safe_float(value_mean),
            )

        for idx, value_std in enumerate(feature_stds):
            self.logger.record(
                f"{prefix}_feature_{idx:02d}_std",
                _safe_float(value_std),
            )

        for idx, value_min in enumerate(feature_mins):
            self.logger.record(
                f"{prefix}_feature_{idx:02d}_min",
                _safe_float(value_min),
            )

        for idx, value_max in enumerate(feature_maxs):
            self.logger.record(
                f"{prefix}_feature_{idx:02d}_max",
                _safe_float(value_max),
            )


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


def make_env(context_name) -> OneShotEnv:
    """Create a training environment for one context."""
    context = make_context(context_name)

    return OneShotEnv(
        action_manager=FlexibleActionManager(context=context),
        observation_manager=MyObservationManager(context=context, continuous=True),  # type: ignore
        reward_function=MyRewardFunction(context=context),
        context=context,
        extra_checks=False,
    )


def try_a_model(
    model,
    context_name: str,
):
    """Run a single simulation with one trained model."""
    context = make_context(context_name)

    world, _ = context.generate(
        types=(OneShotRLAgent,),
        params=(
            dict(
                models=[model_wrapper(model)],
                observation_managers=[MyObservationManager(context, continuous=True)],
                action_managers=[FlexibleActionManager(context)],
            ),
        ),
    )

    world.run_with_progress()
    return world


def train_one(context_name, ntrain, params, queue):
    """Train one model for one context."""
    print(f"Training as {context_name}")
    env = None

    run_name = os.environ.get("RUN_NAME", "default")
    eval_freq = int(os.environ.get("EVAL_FREQ", str(max(ntrain // 5, 1))))
    n_eval_episodes = int(os.environ.get("N_EVAL_EPISODES", "3"))
    diagnostics_freq = int(os.environ.get("DIAGNOSTICS_FREQ", str(max(ntrain // 20, 1))))

    callbacks: list[BaseCallback] = [
        ProgressCallback(queue, context_name),
        TrainingDiagnosticsCallback(log_freq=diagnostics_freq),
    ]

    if eval_freq > 0 and n_eval_episodes > 0:
        callbacks.append(
            EvaluationCallback(
                context_name=context_name,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
            )
        )

    try:
        env = VecMonitor(
            SubprocVecEnv(
                [
                    lambda context_name=context_name: make_env(context_name)
                    for _ in range(params["n_envs"])
                ]
            )
        )

        model = TrainingAlgorithm(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=f"./tensorboard_logs/{run_name}/{context_name}",
        )

        model.learn(
            total_timesteps=ntrain,
            progress_bar=False,
            callback=callbacks,
            tb_log_name=context_name,
        )

        model_path = MODEL_PATH.parent / f"{MODEL_PATH.name}{context_name}"
        model.save(model_path)

    finally:
        if env is not None:
            env.close()

        queue.put((context_name, None))


def main(ntrain: int = NTRAINING):
    """Train models for selected contexts."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    if slurm_cpus:
        total_cores = int(slurm_cpus)
    else:
        total_cores = os.cpu_count() or 1

    n_parallel = min(len(CONTEXTS), max(1, (total_cores - 2) // 2), 3)
    params = get_parallelization_params(n_models_parallel=n_parallel)

    print("=== Training config ===")
    print(f"ntrain: {ntrain}")
    print(f"contexts: {CONTEXTS}")
    print(f"run_name: {os.environ.get('RUN_NAME', 'default')}")
    print(f"eval_freq: {os.environ.get('EVAL_FREQ', f'{max(ntrain // 5, 1)}')}")
    print(f"n_eval_episodes: {os.environ.get('N_EVAL_EPISODES', '3')}")
    print(f"diagnostics_freq: {os.environ.get('DIAGNOSTICS_FREQ', f'{max(ntrain // 20, 1)}')}")
    print(f"rl_agent_code: {_rl_agent_code()}")

    queue = Queue()

    for i in range(0, len(CONTEXTS), n_parallel):
        batch = CONTEXTS[i : i + n_parallel]

        bars = {
            name: tqdm(total=ntrain, desc=name, position=j, leave=True)
            for j, name in enumerate(batch)
        }

        processes = [
            Process(target=train_one, args=(context_name, ntrain, params, queue))
            for context_name in batch
        ]

        for process in processes:
            process.start()

        finished = 0

        while finished < len(batch):
            context_name, steps = queue.get()

            if steps is None:
                bars[context_name].close()
                finished += 1
            else:
                bars[context_name].update(steps)

        for process in processes:
            process.join()


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else NTRAINING)