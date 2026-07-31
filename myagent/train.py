# trains an RL model
import atexit
import csv
import glob
import json
import logging
import os
import random
import re
from multiprocessing import Process, Queue
from queue import Empty as QueueEmpty
from typing import Any
from pathlib import Path

import numpy as np
from negmas.sao import SAOResponse, ResponseType
from rich import print
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.context import GeneralContext
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.reward import RewardFunction

from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from .action import make_action_manager
from .opponents import describe_pool, resolve_opponent_pool
from .common import (
    LOG_ROOT,
    MODEL_PATH,
    CONTEXTS,
    MyObservationManager,
    TrainingAlgorithm,
    get_parallelization_params,
    make_context,
)

NTRAINING = 300000  # number of training steps. Can also be passed as an argument.


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


def evaluate_model(
    model, context_name: str, seed: int | None = None
) -> dict[str, float]:
    """Run one evaluation world and return loggable metrics.

    If ``seed`` is given, the world generation and opponent randomness are made
    reproducible by seeding the global RNG, so the same eval world is used at
    every evaluation point during training.
    """
    context = make_context(context_name)

    np_state = np.random.get_state() if seed is not None else None
    py_state = random.getstate() if seed is not None else None
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    try:
        world, _ = context.generate(
            types=(OneShotRLAgent,),
            params=(
                dict(
                    models=[model_wrapper(model, deterministic=True)],
                    observation_managers=[MyObservationManager(context, continuous=True)],
                    action_managers=[make_action_manager(context)],
                ),
            ),
        )

        if hasattr(world, "run"):
            world.run()
        else:
            world.run_with_progress()
    finally:
        # Restore the RNG so training randomness is unaffected by eval seeding.
        if seed is not None:
            np.random.set_state(np_state)  # type: ignore[arg-type]
            random.setstate(py_state)  # type: ignore[arg-type]

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


def _diff_deals(
    cur_qty: dict, cur_cost: dict, prev_qty: dict, prev_cost: dict
) -> list[tuple[float, int]]:
    """(unit_price, quantity) for deals realized since a ``before_action`` snapshot.

    Reads per-partner secured quantity and total price (``awi.sales`` /
    ``awi.sales_cost`` for selling, ``awi.supplies`` / ``awi.supplies_cost`` for
    buying) and returns only *positive* increments.
    """
    deals: list[tuple[float, int]] = []
    try:
        for partner, qty in cur_qty.items():
            dq = int(qty) - int(prev_qty.get(partner, 0))
            if dq > 0:
                dc = float(cur_cost.get(partner, 0.0)) - float(prev_cost.get(partner, 0.0))
                unit = dc / dq if dq else 0.0
                deals.append((float(unit), dq))
    except Exception:
        return []
    return deals

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


class CheckpointCallback(BaseCallback):
    """Save step-tagged model checkpoints during training."""

    def __init__(self, base_path, save_freq: int):
        super().__init__()
        self.base_path = str(base_path)
        self.save_freq = max(1, save_freq)
        # Start from the current step so a resumed run doesn't immediately
        # re-save the checkpoint it was loaded from.
        self.last_save_step: int | None = None

    def _on_step(self) -> bool:
        if self.last_save_step is None:
            self.last_save_step = self.num_timesteps
            return True
        if self.num_timesteps - self.last_save_step < self.save_freq:
            return True
        self.last_save_step = self.num_timesteps
        try:
            self.model.save(f"{self.base_path}_ckpt{self.num_timesteps}")
        except Exception as e:
            # Checkpointing must never stop training.
            print(f"[checkpoint failed] {self.base_path}: {e}")
        return True


class EvaluationCallback(BaseCallback):
    """Log evaluation metrics to TensorBoard and keep the best-eval model.

    If ``best_path`` is given, the model is saved there whenever the mean
    ``score`` over the fixed eval worlds sets a new record, with a sidecar
    ``<best_path>_meta.json`` recording the score/step. The sidecar is read
    back on construction so a resumed run keeps the historical record instead
    of overwriting the best model with a worse one."""

    # Headline metrics surfaced in their own dashboard category. The "0_" prefix
    # sorts this group to the top in TensorBoard (categories are ordered
    # alphabetically by the tag prefix before the first "/").
    KEY_METRICS = (
        "score",
        "my_score",
        "my_rank",
        "score_gap",
        "score_gap_vs_best_opponent",
        "opponent_score_mean",
        "best_opponent_score",
    )

    def __init__(
        self,
        context_name: str,
        eval_freq: int,
        n_eval_episodes: int,
        best_path: str | None = None,
    ):
        super().__init__()
        self.context_name = context_name
        self.eval_freq = max(1, eval_freq)
        self.n_eval_episodes = max(1, n_eval_episodes)
        self.last_eval_step = 0

        self.best_path = str(best_path) if best_path else None
        self.best_score = -np.inf
        self.best_step = 0
        if self.best_path and os.path.exists(f"{self.best_path}_meta.json"):
            try:
                with open(f"{self.best_path}_meta.json") as f:
                    meta = json.load(f)
                self.best_score = float(meta.get("score", -np.inf))
                self.best_step = int(meta.get("step", 0))
                print(
                    f"[best-model] {self.context_name}: resuming record "
                    f"{self.best_score:.4f} @ {self.best_step}"
                )
            except Exception:
                pass

    def _maybe_save_best(self, score: float) -> None:
        if self.best_path is None or score <= self.best_score:
            return
        self.best_score = score
        self.best_step = self.num_timesteps
        try:
            self.model.save(self.best_path)
            with open(f"{self.best_path}_meta.json", "w") as f:
                json.dump(
                    {"score": self.best_score, "step": self.best_step}, f
                )
        except Exception as e:
            print(f"[best-model save failed] {self.context_name}: {e}")

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True

        self.last_eval_step = self.num_timesteps

        try:
            results = [
                evaluate_model(self.model, self.context_name, seed=i)
                for i in range(self.n_eval_episodes)
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

                if metric_name == "score":
                    self._maybe_save_best(mean_value)

                # Surface the most important metrics in a top-sorted category.
                if metric_name in self.KEY_METRICS:
                    self.logger.record(f"0_key/{metric_name}", mean_value)

                self.logger.record(f"eval/{metric_name}", mean_value)
                self.logger.record(f"eval/{metric_name}_mean", mean_value)
                self.logger.record(f"eval/{metric_name}_std", float(np.std(values)))
                self.logger.record(f"eval/{metric_name}_min", float(np.min(values)))
                self.logger.record(f"eval/{metric_name}_max", float(np.max(values)))

            if self.best_path is not None and np.isfinite(self.best_score):
                self.logger.record("0_key/best_score_so_far", float(self.best_score))

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


# Per-context default shaping weights. Keyed by context class name.
# A matching REWARD_* environment variable always overrides the value here,
# so sweeps that set the env vars explicitly are unaffected;
_CONTEXT_DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {}

# Discount used by both PPO and PBRS. They need to be identical.
GAMMA = 0.95

# Maps each reward-weight attribute to its environment variable and global
# default. Single source of truth for both MyRewardFunction and the config dump.
_REWARD_WEIGHT_ENV: dict[str, tuple[str, float]] = {
    "score_delta_weight": ("REWARD_SCORE_DELTA_WEIGHT", 100.0),
    "need_weight": ("REWARD_NEED_WEIGHT", 0.0),
    "shortfall_weight": ("REWARD_SHORTFALL_WEIGHT", 0.0),
    "overshoot_weight": ("REWARD_OVERSHOOT_WEIGHT", 0.0),
    "disposal_weight": ("REWARD_DISPOSAL_WEIGHT", 0.0),
    "productivity_weight": ("REWARD_PRODUCTIVITY_WEIGHT", 0.0),
    "time_pressure_weight": ("REWARD_TIME_PRESSURE_WEIGHT", 1.0),
    "need_normalizer": ("REWARD_NEED_NORMALIZER", 0.0),
    "price_weight": ("REWARD_PRICE_WEIGHT", 0.0),
    "deal_weight": ("REWARD_DEAL_WEIGHT", 0.0),
    "engagement_weight": ("REWARD_ENGAGEMENT_WEIGHT", 0.0),
    "margin_weight": ("REWARD_MARGIN_WEIGHT", 0.0),
    # Potential-based shaping: policy-invariant, densifies the sparse profit signal without changing the optimum
    "potential_weight": ("REWARD_POTENTIAL_WEIGHT", 0.0),
}


def resolve_reward_weights(context_name: str) -> dict[str, float]:
    """Resolve all reward weights for a context.

    Precedence: environment variable (if set) > per-context default table >
    global default.
    """
    context_defaults = _CONTEXT_DEFAULT_WEIGHTS.get(context_name, {})
    resolved: dict[str, float] = {}

    for attr, (env_key, global_default) in _REWARD_WEIGHT_ENV.items():
        if env_key in os.environ:
            resolved[attr] = float(os.environ[env_key])
        else:
            resolved[attr] = float(context_defaults.get(attr, global_default))

    return resolved


class MyRewardFunction(RewardFunction):
    """Reward shaping with configurable terms."""

    def __init__(self, context: GeneralContext):
        super().__init__()
        self.context = context

        weights = resolve_reward_weights(type(context).__name__)

        self.score_delta_weight = weights["score_delta_weight"]
        self.need_weight = weights["need_weight"]
        self.shortfall_weight = weights["shortfall_weight"]
        self.overshoot_weight = weights["overshoot_weight"]
        self.disposal_weight = weights["disposal_weight"]
        self.productivity_weight = weights["productivity_weight"]
        self.time_pressure_weight = weights["time_pressure_weight"]
        self.need_normalizer = weights["need_normalizer"]

        # Context-specific shaping weights. Side (buy/sell) is auto-detected from the context.
        self.price_weight = weights["price_weight"]
        self.deal_weight = weights["deal_weight"]
        self.engagement_weight = weights["engagement_weight"]
        self.margin_weight = weights["margin_weight"]
        self.potential_weight = weights["potential_weight"]
        # Which potential Φ the PBRS term uses (only matters when potential_weight != 0)
        self.potential_kind = (
            os.environ.get("REWARD_POTENTIAL_KIND", "coverage").strip().lower()
        )
        if self.potential_kind not in ("coverage", "dayprofit"):
            raise ValueError(
                f"REWARD_POTENTIAL_KIND must be 'coverage' or 'dayprofit', "
                f"got {self.potential_kind!r}"
            )

        self.log_reward_components = (
            os.environ.get("LOG_REWARD_COMPONENTS", "1") != "0"
        )

        self.reward_log_file = None
        self.reward_log_writer = None
        self.reward_log_rows_since_flush = 0
        self.reward_log_flush_every = int(
            os.environ.get("REWARD_LOG_FLUSH_EVERY", "1000")
        )

        if self.log_reward_components:
            self._open_reward_log()

        atexit.register(self._close_reward_log)

    def _potential(self, awi: OneShotAWI) -> float:
        """Potential Φ(s) for potential-based reward shaping (PBRS)."""
        if self.potential_weight == 0.0:
            return 0.0
        if self.potential_kind == "dayprofit":
            return self._potential_dayprofit(awi)
        return self._potential_coverage(awi)

    def _potential_dayprofit(self, awi: OneShotAWI) -> float:
        """Φ = realized profit of the current day so far (normalized)."""
        try:
            is_consumer = "Consumer" in type(self.context).__name__
            catalog_in, catalog_out = _catalog_prices(awi)
            prod_cost = _safe_float(
                getattr(getattr(awi, "profile", None), "cost", 0.0)
            )
            n_lines = max(1.0, _safe_float(getattr(awi, "n_lines", 1.0), default=1.0))
            if is_consumer:
                qty = getattr(awi, "supplies", {}) or {}
                value = getattr(awi, "supplies_cost", {}) or {}
                total_qty = float(sum(qty.values()))
                total_spend = float(sum(value.values()))
                profit = (catalog_out - prod_cost) * total_qty - total_spend
                price_scale = max(catalog_in, 1e-6)
            else:
                qty = getattr(awi, "sales", {}) or {}
                value = getattr(awi, "sales_cost", {}) or {}
                total_qty = float(sum(qty.values()))
                total_revenue = float(sum(value.values()))
                profit = total_revenue - (catalog_in + prod_cost) * total_qty
                price_scale = max(catalog_out, 1e-6)
            normalized = profit / (price_scale * n_lines)
            return self.potential_weight * float(np.clip(normalized, -2.0, 2.0))
        except Exception:
            return 0.0

    def _potential_coverage(self, awi: OneShotAWI) -> float:
        """Φ = coverage ∈ [0, 1]: fraction of the *active need* already
        secured (1 = fully covered). Higher Φ = less expected
        shortfall/disposal.
        """
        try:
            needed_sales = _safe_float(getattr(awi, "needed_sales", 0.0))
            needed_supplies = _safe_float(getattr(awi, "needed_supplies", 0.0))
            name = type(self.context).__name__
            if "Supplier" in name:
                active_need = needed_sales
            elif "Consumer" in name:
                active_need = needed_supplies
            else:
                active_need = (
                    needed_sales
                    if abs(needed_sales) >= abs(needed_supplies)
                    else needed_supplies
                )
            unmet = max(0.0, active_need)
            scale = (
                self.need_normalizer
                if self.need_normalizer > 0.0
                else max(1.0, _safe_float(getattr(awi, "n_lines", 1.0), default=1.0))
            )
            coverage = 1.0 - min(1.0, unmet / max(1.0, scale))
            return self.potential_weight * coverage
        except Exception:
            return 0.0

    def before_action(self, awi: OneShotAWI) -> dict[str, Any]:
        # Snapshot the score, per-partner secured quantity/price, and the PBRS
        # potential Φ(s)
        return {
            "score": float(getattr(awi, "current_score", 0.0)),
            "sales": dict(getattr(awi, "sales", {}) or {}),
            "sales_cost": dict(getattr(awi, "sales_cost", {}) or {}),
            "supplies": dict(getattr(awi, "supplies", {}) or {}),
            "supplies_cost": dict(getattr(awi, "supplies_cost", {}) or {}),
            "potential": self._potential(awi),
        }

    def __call__(self, awi: OneShotAWI, action: dict[str, SAOResponse], info: Any):
        before = info if isinstance(info, dict) else {}
        previous_score = _safe_float(before.get("score", info))
        current_score = _safe_float(getattr(awi, "current_score", previous_score))
        score_delta = current_score - previous_score
        score_delta_bonus = self.score_delta_weight * score_delta

        # Deals realized this step, read from a reward-time-valid source.
        sell_deals = _diff_deals(
            getattr(awi, "sales", {}) or {},
            getattr(awi, "sales_cost", {}) or {},
            before.get("sales", {}),
            before.get("sales_cost", {}),
        )
        buy_deals = _diff_deals(
            getattr(awi, "supplies", {}) or {},
            getattr(awi, "supplies_cost", {}) or {},
            before.get("supplies", {}),
            before.get("supplies_cost", {}),
        )

        reward_terms = self._calculate_reward_terms(awi)
        context_terms = self._calculate_context_shaping(
            awi, action, sell_deals, buy_deals
        )

        # Potential-based shaping: γ·Φ(s') − Φ(s). Policy-invariant.
        prev_potential = _safe_float(before.get("potential", 0.0))
        cur_potential = self._potential(awi)
        pbrs_bonus = GAMMA * cur_potential - prev_potential

        final_reward = (
            score_delta_bonus
            + reward_terms["need_penalty"] # type: ignore
            + reward_terms["shortfall_penalty_term"]
            + reward_terms["overshoot_penalty"]
            + reward_terms["disposal_penalty_term"]
            + reward_terms["productivity_bonus"]
            + context_terms["margin_bonus"]
            + context_terms["price_bonus"]
            + context_terms["deal_bonus"]
            + context_terms["engagement_bonus"]
            + pbrs_bonus
        )

        if self.log_reward_components:
            self._log_reward_components(
                awi=awi,
                action=action,
                previous_score=previous_score,
                current_score=current_score,
                score_delta=score_delta,
                score_delta_bonus=score_delta_bonus,
                final_reward=final_reward,
                reward_terms=reward_terms,
                context_terms=context_terms,
                pbrs_bonus=pbrs_bonus,
            )

        return final_reward

    def _calculate_reward_terms(self, awi: OneShotAWI) -> dict[str, float | str]:
        """Calculate configurable reward shaping terms."""
        needed_sales = _safe_float(getattr(awi, "needed_sales", 0.0))
        needed_supplies = _safe_float(getattr(awi, "needed_supplies", 0.0))

        context_name = type(self.context).__name__

        if "Supplier" in context_name:
            active_need = needed_sales
            active_need_type = "sales"
        elif "Consumer" in context_name:
            active_need = needed_supplies
            active_need_type = "supplies"
        else:
            if abs(needed_sales) >= abs(needed_supplies):
                active_need = needed_sales
                active_need_type = "sales"
            else:
                active_need = needed_supplies
                active_need_type = "supplies"

        unmet_need = max(0.0, active_need)
        overshoot = max(0.0, -active_need)

        n_lines = max(1.0, _safe_float(getattr(awi, "n_lines", 1.0), default=1.0))

        if self.need_normalizer > 0.0:
            need_scale = self.need_normalizer
        else:
            need_scale = n_lines

        unmet_need_scaled = unmet_need / max(1.0, need_scale)
        overshoot_scaled = overshoot / max(1.0, need_scale)

        relative_time = _safe_float(getattr(awi, "relative_time", 0.0))
        time_multiplier = 1.0 + self.time_pressure_weight * relative_time

        current_shortfall_penalty = _safe_float(
            getattr(awi, "current_shortfall_penalty", 0.0)
        )
        current_disposal_cost = _safe_float(
            getattr(awi, "current_disposal_cost", 0.0)
        )

        need_penalty = -self.need_weight * unmet_need_scaled * time_multiplier
        shortfall_penalty_term = (
            -self.shortfall_weight
            * unmet_need_scaled
            * current_shortfall_penalty
            * time_multiplier
        )

        overshoot_penalty = -self.overshoot_weight * overshoot_scaled
        disposal_penalty_term = (
            -self.disposal_weight
            * overshoot_scaled
            * current_disposal_cost
        )

        productivity_proxy = max(0.0, 1.0 - min(1.0, unmet_need_scaled))
        productivity_bonus = self.productivity_weight * productivity_proxy

        return {
            "active_need_type": active_need_type,
            "active_need": active_need,
            "needed_sales": needed_sales,
            "needed_supplies": needed_supplies,
            "unmet_need": unmet_need,
            "overshoot": overshoot,
            "need_scale": need_scale,
            "unmet_need_scaled": unmet_need_scaled,
            "overshoot_scaled": overshoot_scaled,
            "relative_time": relative_time,
            "time_multiplier": time_multiplier,
            "current_shortfall_penalty": current_shortfall_penalty,
            "current_disposal_cost": current_disposal_cost,
            "productivity_proxy": productivity_proxy,
            "need_penalty": need_penalty,
            "shortfall_penalty_term": shortfall_penalty_term,
            "overshoot_penalty": overshoot_penalty,
            "disposal_penalty_term": disposal_penalty_term,
            "productivity_bonus": productivity_bonus,
        }

    def _calculate_context_shaping(
        self,
        awi: OneShotAWI,
        action: dict[str, SAOResponse],
        sell_deals: list[tuple[float, int]],
        buy_deals: list[tuple[float, int]],
    ) -> dict[str, float]:
        """Side-aware price/deal/engagement shaping.

        ``sell_deals`` / ``buy_deals`` are ``(unit_price, quantity)`` pairs for
        deals *actually realized this step* (from ``_diff_deals``). The trading
        *side* and the "good price" direction are auto-detected from the context:

        - ``margin_weight``: profit-aligned reward for realized per-deal margin
          (sell price minus cost basis, or value minus buy price), normalized by
          price scale and capacity. This is the primary dense signal.
        - ``price_weight``: quantity-weighted reward for deals that beat the
          catalog price (above catalog when selling, below when buying).
        - ``deal_weight``: reward for realized volume, normalized by capacity.
        - ``engagement_weight``: reward for not ending/abandoning negotiations.
        """
        terms = {
            "margin_bonus": 0.0,
            "price_bonus": 0.0,
            "deal_bonus": 0.0,
            "engagement_bonus": 0.0,
        }

        try:
            is_consumer = "Consumer" in type(self.context).__name__
            catalog_in, catalog_out = _catalog_prices(awi)
            prod_cost = _safe_float(getattr(getattr(awi, "profile", None), "cost", 0.0))
            n_lines = max(1.0, _safe_float(getattr(awi, "n_lines", 1.0), default=1.0))

            if is_consumer:
                states = getattr(awi, "current_buy_states", {}) or {}
                deals = buy_deals
                reference = catalog_in
                # Buying below catalog is favorable.
                sign = -1.0
            else:
                states = getattr(awi, "current_sell_states", {}) or {}
                deals = sell_deals
                reference = catalog_out
                # Selling above catalog is favorable.
                sign = 1.0


            if deals and self.margin_weight != 0.0:
                if is_consumer:
                    # value of one produced unit vs the price paid to acquire input
                    break_even = catalog_out - prod_cost
                    price_scale = max(catalog_in, 1e-6)
                    unit_margins = [(break_even - price, qty) for price, qty in deals]
                else:
                    # sale price vs the cost basis of producing one output unit
                    break_even = catalog_in + prod_cost
                    price_scale = max(catalog_out, 1e-6)
                    unit_margins = [(price - break_even, qty) for price, qty in deals]
                margin_bonus = 0.0
                for margin, qty in unit_margins:
                    margin_bonus += (
                        float(np.clip(margin / price_scale, -0.5, 0.5))
                        * (qty / n_lines)
                    )
                terms["margin_bonus"] = self.margin_weight * margin_bonus

            if deals and self.price_weight != 0.0:
                total_qty = sum(q for _, q in deals)
                price_bonus = 0.0
                for price, qty in deals:
                    ratio = sign * (price - reference) / max(reference, 1e-6)
                    price_bonus += (
                        float(np.clip(ratio, -0.25, 0.25)) * (qty / max(total_qty, 1))
                    )
                terms["price_bonus"] = self.price_weight * price_bonus

            if deals and self.deal_weight != 0.0:
                realized_qty = sum(q for _, q in deals)
                n_lines = max(1.0, _safe_float(getattr(awi, "n_lines", 1.0), default=1.0))
                terms["deal_bonus"] = self.deal_weight * float(
                    np.clip(realized_qty / n_lines, 0.0, 1.0)
                )

            partners = set(states.keys())
            if partners and self.engagement_weight != 0.0:
                engaged = sum(
                    1
                    for pid, response in action.items()
                    if pid in partners
                    and response.response
                    not in (
                        ResponseType.END_NEGOTIATION,
                        ResponseType.NO_RESPONSE,
                        ResponseType.WAIT,
                    )
                )
                terms["engagement_bonus"] = self.engagement_weight * (
                    engaged / max(len(partners), 1)
                )
        except Exception:
            return {
                "margin_bonus": 0.0,
                "price_bonus": 0.0,
                "deal_bonus": 0.0,
                "engagement_bonus": 0.0,
            }

        return terms

    def _open_reward_log(self) -> None:
        """Open one reward component log per worker process."""
        run_name = os.environ.get("RUN_NAME", "default")
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        context_name = type(self.context).__name__
        pid = os.getpid()

        log_dir = os.path.join(
            LOG_ROOT,
            "reward_component_logs",
            run_name,
            context_name,
            job_id,
        )
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, f"reward_components_{pid}.csv")

        self.reward_log_file = open(log_path, "w", newline="")
        self.reward_log_writer = csv.DictWriter(
            self.reward_log_file,
            fieldnames=[
                "pid",
                "context",
                "current_step",
                "relative_time",
                "n_steps",
                "active_need_type",
                "active_need",
                "needed_sales",
                "needed_supplies",
                "unmet_need",
                "overshoot",
                "need_scale",
                "unmet_need_scaled",
                "overshoot_scaled",
                "time_multiplier",
                "current_score",
                "previous_score",
                "score_delta",
                "score_delta_weight",
                "score_delta_bonus",
                "need_weight",
                "need_penalty",
                "shortfall_weight",
                "shortfall_penalty_term",
                "overshoot_weight",
                "overshoot_penalty",
                "disposal_weight",
                "disposal_penalty_term",
                "productivity_weight",
                "productivity_proxy",
                "productivity_bonus",
                "margin_weight",
                "margin_bonus",
                "potential_weight",
                "potential_kind",
                "pbrs_bonus",
                "price_weight",
                "price_bonus",
                "deal_weight",
                "deal_bonus",
                "engagement_weight",
                "engagement_bonus",
                "final_reward",
                "current_disposal_cost",
                "current_shortfall_penalty",
                "current_storage_cost",
                "current_inventory_input",
                "current_inventory_output",
                "n_action_responses",
                "n_action_accept",
                "n_action_reject",
                "n_action_end",
                "n_action_offer",
            ],
        )
        self.reward_log_writer.writeheader()

    def _close_reward_log(self) -> None:
        """Flush and close the reward component log."""
        if self.reward_log_file is None:
            return

        try:
            self.reward_log_file.flush()
            self.reward_log_file.close()
        except Exception:
            pass

        self.reward_log_file = None
        self.reward_log_writer = None

    def _log_reward_components(
        self,
        awi: OneShotAWI,
        action: dict[str, SAOResponse],
        previous_score: float,
        current_score: float,
        score_delta: float,
        score_delta_bonus: float,
        final_reward: float,
        reward_terms: dict[str, float | str],
        context_terms: dict[str, float],
        pbrs_bonus: float = 0.0,
    ) -> None:
        """Write one reward component row."""
        if self.reward_log_writer is None:
            return

        action_counts = self._summarize_action(action)

        self.reward_log_writer.writerow(
            {
                "pid": os.getpid(),
                "context": type(self.context).__name__,
                "current_step": _safe_float(getattr(awi, "current_step", 0)),
                "relative_time": reward_terms["relative_time"],
                "n_steps": _safe_float(getattr(awi, "n_steps", 0)),
                "active_need_type": reward_terms["active_need_type"],
                "active_need": reward_terms["active_need"],
                "needed_sales": reward_terms["needed_sales"],
                "needed_supplies": reward_terms["needed_supplies"],
                "unmet_need": reward_terms["unmet_need"],
                "overshoot": reward_terms["overshoot"],
                "need_scale": reward_terms["need_scale"],
                "unmet_need_scaled": reward_terms["unmet_need_scaled"],
                "overshoot_scaled": reward_terms["overshoot_scaled"],
                "time_multiplier": reward_terms["time_multiplier"],
                "current_score": current_score,
                "previous_score": previous_score,
                "score_delta": score_delta,
                "score_delta_weight": self.score_delta_weight,
                "score_delta_bonus": score_delta_bonus,
                "need_weight": self.need_weight,
                "need_penalty": reward_terms["need_penalty"],
                "shortfall_weight": self.shortfall_weight,
                "shortfall_penalty_term": reward_terms["shortfall_penalty_term"],
                "overshoot_weight": self.overshoot_weight,
                "overshoot_penalty": reward_terms["overshoot_penalty"],
                "disposal_weight": self.disposal_weight,
                "disposal_penalty_term": reward_terms["disposal_penalty_term"],
                "productivity_weight": self.productivity_weight,
                "productivity_proxy": reward_terms["productivity_proxy"],
                "productivity_bonus": reward_terms["productivity_bonus"],
                "margin_weight": self.margin_weight,
                "margin_bonus": context_terms["margin_bonus"],
                "potential_weight": self.potential_weight,
                "potential_kind": self.potential_kind,
                "pbrs_bonus": pbrs_bonus,
                "price_weight": self.price_weight,
                "price_bonus": context_terms["price_bonus"],
                "deal_weight": self.deal_weight,
                "deal_bonus": context_terms["deal_bonus"],
                "engagement_weight": self.engagement_weight,
                "engagement_bonus": context_terms["engagement_bonus"],
                "final_reward": final_reward,
                "current_disposal_cost": reward_terms["current_disposal_cost"],
                "current_shortfall_penalty": reward_terms["current_shortfall_penalty"],
                "current_storage_cost": _safe_float(
                    getattr(awi, "current_storage_cost", 0.0)
                ),
                "current_inventory_input": _safe_float(
                    getattr(awi, "current_inventory_input", 0.0)
                ),
                "current_inventory_output": _safe_float(
                    getattr(awi, "current_inventory_output", 0.0)
                ),
                **action_counts,
            }
        )

        self.reward_log_rows_since_flush += 1

        if self.reward_log_rows_since_flush >= self.reward_log_flush_every:
            self.reward_log_file.flush()  # type: ignore
            self.reward_log_rows_since_flush = 0

    def _summarize_action(self, action: dict[str, SAOResponse]) -> dict[str, float]:
        """Summarize response types in the selected action."""
        counts = {
            "n_action_responses": 0.0,
            "n_action_accept": 0.0,
            "n_action_reject": 0.0,
            "n_action_end": 0.0,
            "n_action_offer": 0.0,
        }

        if not isinstance(action, dict):
            return counts

        counts["n_action_responses"] = float(len(action))

        for response in action.values():
            resp = getattr(response, "response", None)

            outcome = getattr(response, "outcome", None)

            if resp == ResponseType.ACCEPT_OFFER:
                counts["n_action_accept"] += 1.0
            elif resp == ResponseType.REJECT_OFFER:
                counts["n_action_reject"] += 1.0
            elif resp == ResponseType.END_NEGOTIATION:
                counts["n_action_end"] += 1.0

            if outcome is not None:
                counts["n_action_offer"] += 1.0

        return counts


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

def make_env(context_name, log: bool | None = None) -> OneShotEnv:
    # When `log` is not passed explicitly, fall back to the LOG_WORLD env var
    # ("1" enables the verbose fail-fast diagnostic profile; default off).
    if log is None:
        log = os.environ.get("LOG_WORLD", "0") != "0"

    # World construction params. These are passed to the world via the
    # context's `world_params`
    if log:
        world_params: dict[str, Any] = dict(
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
    else:
        world_params = dict(
            debug=False,
            ignore_agent_exceptions=True,
            ignore_negotiation_exceptions=True,
            ignore_contract_execution_exceptions=True,
            ignore_simulation_exceptions=True,
        )

    # OPPONENT_POOL=strong mixes top-2024 qualifiers into the non-competitor
    # slots (myagent/opponents.py). TRAINING worlds only — evaluate_model and
    # deployment keep the default pool so eval curves stay comparable.
    context = make_context(context_name, non_competitors=resolve_opponent_pool())
    context.world_params.update(world_params)

    return OneShotEnv(
        action_manager=make_action_manager(context),
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
                models=[model_wrapper(model, deterministic=True)],
                observation_managers=[MyObservationManager(context, continuous=True)],
                action_managers=[make_action_manager(context)],
            ),
        ),
    )

    world.run_with_progress()
    return world

def try_a_trained_model(context_name: str):
    """Runs a simulation with one agent controlled by the already trained model for the given context"""
    path = MODEL_PATH.parent / f"{MODEL_PATH.name}{context_name}"
    model = TrainingAlgorithm.load(path)
    context = make_context(context_name)
    world, agent = context.generate(
        types=(OneShotRLAgent,),
        params=(
            dict(
                models=[model_wrapper(model, deterministic=True)],
                observation_managers=[MyObservationManager(context, continuous=True)],
                action_managers=[make_action_manager(context)],
            ),
        ),
    )

    aid = agent[0].id
    print(f"Assigned Agent: {agent[0].name}, {aid}")
    world.run_with_progress()

    scores = world.scores()
    print(scores)
    print(f"Our Score: {scores[aid]}")
    print(f"Agreement: {world.agreement_rate}")
    print(f"Bankrupt: {world.is_bankrupt[aid]}")
    print(f"neg_requests_received: {world.neg_requests_received[aid]}")
    print(f"neg_requests_rejected: {world.neg_requests_rejected[aid]}")
    print(f"neg_requests_sent: {world.neg_requests_sent[aid]}")
    print(f"negs_initiated: {world.negs_initiated[aid]}")
    print(f"negs_failed: {world.negs_failed[aid]}")


    return world


def _newest_checkpoint(base_path) -> tuple[str, int] | None:
    """Return (path, steps) of the highest-step checkpoint for a model base."""
    best: tuple[str, int] | None = None
    for path in glob.glob(f"{base_path}_ckpt*.zip"):
        m = re.search(r"_ckpt(\d+)\.zip$", path)
        if not m:
            continue
        steps = int(m.group(1))
        if best is None or steps > best[1]:
            best = (path, steps)
    return best


def train_one(context_name, ntrain, params, queue):
    """Train one model for one context."""
    print(f"Training as {context_name}")
    env = None

    run_name = os.environ.get("RUN_NAME", "default")
    eval_freq = int(os.environ.get("EVAL_FREQ", str(max(ntrain // 5, 1))))
    n_eval_episodes = int(os.environ.get("N_EVAL_EPISODES", "3"))
    diagnostics_freq = int(os.environ.get("DIAGNOSTICS_FREQ", str(max(ntrain // 20, 1))))
    checkpoint_freq = int(os.environ.get("CHECKPOINT_FREQ", "100000"))
    resume = os.environ.get("RESUME", "0") != "0"

    seed_str = os.environ.get("SEED")
    seed = int(seed_str) if seed_str not in (None, "") else None

    if seed is not None:
        model_path = (
            MODEL_PATH.parent
            / f"{MODEL_PATH.name}{context_name}_{run_name}_seed{seed}"
        )
    else:
        model_path = MODEL_PATH.parent / f"{MODEL_PATH.name}{context_name}"

    progress_bars = os.environ.get("PROGRESS_BARS", "1") != "0"

    callbacks: list[BaseCallback] = [
        TrainingDiagnosticsCallback(log_freq=diagnostics_freq),
    ]

    if progress_bars:
        callbacks.insert(0, ProgressCallback(queue, context_name))

    if checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(base_path=model_path, save_freq=checkpoint_freq)
        )

    if eval_freq > 0 and n_eval_episodes > 0:
        callbacks.append(
            EvaluationCallback(
                context_name=context_name,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                best_path=f"{model_path}_best",
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

        tensorboard_log = f"./{LOG_ROOT}/tensorboard_logs/{run_name}/{context_name}"

        warm_start = os.environ.get("WARM_START", "0") != "0"
        warm_start_strict = os.environ.get("WARM_START_STRICT", "1") != "0"
        warm_start_reset_timesteps = (
            os.environ.get("WARM_START_RESET_TIMESTEPS", "1") != "0"
        )

        warm_start_model_dir = Path(
            os.environ.get("WARM_START_MODEL_DIR", str(MODEL_PATH.parent))
        )
        warm_start_path = warm_start_model_dir / f"{MODEL_PATH.name}{context_name}"
        warm_start_zip_path = warm_start_path.with_suffix(".zip")
        warm_start_loaded = False

        checkpoint = _newest_checkpoint(model_path) if resume else None

        if checkpoint is not None:
            checkpoint_path, checkpoint_steps = checkpoint
            model = TrainingAlgorithm.load(checkpoint_path, env=env)
            model.tensorboard_log = tensorboard_log
            remaining = max(0, ntrain - model.num_timesteps)

            print(
                f"[resume] {context_name}: loaded {checkpoint_path} "
                f"({model.num_timesteps} steps done, {remaining} to go)"
            )

            if remaining > 0:
                model.learn(
                    total_timesteps=remaining,
                    progress_bar=False,
                    callback=callbacks,
                    tb_log_name=context_name,
                    reset_num_timesteps=False,
                )

        else:
            if resume:
                print(f"[resume] {context_name}: no checkpoint found, starting fresh")

            policy_kwargs = dict(
                net_arch=[128, 128]
            )

            if warm_start and warm_start_zip_path.exists():
                print(f"Warm starting {context_name} from {warm_start_zip_path}")
                model = TrainingAlgorithm.load(warm_start_path, env=env)
                model.tensorboard_log = tensorboard_log
                warm_start_loaded = True

            else:
                if warm_start:
                    message = (
                        f"Warm start requested, but model not found for "
                        f"{context_name}: {warm_start_zip_path}"
                    )

                    if warm_start_strict:
                        raise FileNotFoundError(message)

                    print(message)
                    print("Training from scratch.")

                model = TrainingAlgorithm(
                    "MlpPolicy",
                    env,
                    verbose=0,
                    policy_kwargs=policy_kwargs,
                    seed=seed,
                    tensorboard_log=tensorboard_log,
                    ent_coef=0.01,
                    gamma=GAMMA,
                    n_steps=512,
                    batch_size=256,
                )  # type: ignore learning_rate must be passed by the algorithm itself

            model.learn(
                total_timesteps=ntrain,
                progress_bar=False,
                callback=callbacks,
                tb_log_name=context_name,
                reset_num_timesteps=not (
                    warm_start_loaded and not warm_start_reset_timesteps
                ),
            )

        model.save(model_path)

    finally:
        if env is not None:
            try:
                env.close()
            except Exception as e:
                print(f"[env close failed] {context_name}: {e}")

        queue.put((context_name, None))

def test_train(context_name):
    env = make_env(context_name)
    obs, _ = env.reset()
    world = env._world
    print(type(world))
    for agent_id, agent in world.agents.items():
        print(agent_id, type(agent).__name__)


def main(ntrain: int = NTRAINING):
    """Train models for selected contexts."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    if slurm_cpus:
        total_cores = int(slurm_cpus)
    else:
        total_cores = os.cpu_count() or 1

    # MAX_PARALLEL_MODELS caps how many context models train at once (each one
    # runs its own learner + up to 8 env workers, ~9 busy cores). Default 3
    # matches the historical 48-CPU allocations (6 contexts = 2 batches); on a
    # 64-CPU node MAX_PARALLEL_MODELS=6 trains all contexts in one batch.
    max_parallel = int(os.environ.get("MAX_PARALLEL_MODELS", "3"))
    n_parallel = min(len(CONTEXTS), max(1, (total_cores - 2) // 2), max(1, max_parallel))

    params = get_parallelization_params(n_models_parallel=n_parallel)

    print("=== Training config ===")
    print(f"ntrain: {ntrain}")
    print(f"contexts: {CONTEXTS}")
    print(f"run_name: {os.environ.get('RUN_NAME', 'default')}")
    print(f"seed: {os.environ.get('SEED', 'None (nondeterministic)')}")
    print(f"diagnostics_freq: {os.environ.get('DIAGNOSTICS_FREQ', f'{max(ntrain // 20, 1)}')}")
    print(f"checkpoint_freq: {os.environ.get('CHECKPOINT_FREQ', '100000')}")
    print(f"resume: {os.environ.get('RESUME', '0')}")
    print(f"progress_bars: {os.environ.get('PROGRESS_BARS', '1')}")
    print(f"rl_agent_code: {_rl_agent_code()}")
    print(
        f"action_manager: {type(make_action_manager(make_context(CONTEXTS[0]))).__name__} "
        f"(ACTION_MANAGER={os.environ.get('ACTION_MANAGER', 'accept')})"
    )
    log_world = os.environ.get("LOG_WORLD", "0") != "0"
    print(f"log_world: {log_world} ({'debug/fail-fast' if log_world else 'robust'})")
    print(
        f"potential_kind: {os.environ.get('REWARD_POTENTIAL_KIND', 'coverage')} "
        f"(only used when potential_weight != 0)"
    )
    # Resolving here (not just in the workers) fails fast on a bad
    # OPPONENT_POOL before any training process is spawned.
    print(
        f"opponent_pool: {describe_pool(resolve_opponent_pool())} "
        f"(OPPONENT_POOL={os.environ.get('OPPONENT_POOL', 'default')}; training worlds only)"
    )

    warm_start_enabled = os.environ.get("WARM_START", "0") != "0"
    print(f"warm_start: {int(warm_start_enabled)}")
    if warm_start_enabled:
        print(
            "warm_start_model_dir: "
            f"{os.environ.get('WARM_START_MODEL_DIR', str(MODEL_PATH.parent))}"
        )
    print("=== Resolved reward weights (per context) ===")
    for context_name in CONTEXTS:
        weights = resolve_reward_weights(context_name)
        nonzero = {
            attr: value for attr, value in weights.items() if value != 0.0
        }
        print(f"{context_name}: {nonzero}")


    progress_bars = os.environ.get("PROGRESS_BARS", "1") != "0"
    queue = Queue()

    for i in range(0, len(CONTEXTS), n_parallel):
        batch = CONTEXTS[i : i + n_parallel]

        if progress_bars:
            bars = {
                name: tqdm(total=ntrain, desc=name, position=j, leave=True)
                for j, name in enumerate(batch)
            }
        else:
            bars = {}
            print(f"=== batch: {batch} ===")

        processes = [
            Process(target=train_one, args=(context_name, ntrain, params, queue))
            for context_name in batch
        ]

        for process in processes:
            process.start()

        waiting = dict(zip(batch, processes))

        while waiting:
            try:
                context_name, steps = queue.get(timeout=60)
            except QueueEmpty:
                for name, process in list(waiting.items()):
                    if not process.is_alive():
                        process.join()
                        if name in bars:
                            bars[name].close()
                        print(
                            f"=== DIED without finishing: {name} "
                            f"(exitcode {process.exitcode}; -9 = SIGKILL, "
                            f"usually the OOM killer) ==="
                        )
                        del waiting[name]
                continue

            if steps is None:
                if context_name in bars:
                    bars[context_name].close()
                else:
                    print(f"=== finished: {context_name} ===")
                waiting.pop(context_name, None)
            elif context_name in bars:
                bars[context_name].update(steps)

        failed_processes = []

        for context_name, process in zip(batch, processes):
            process.join()

            if process.exitcode != 0:
                failed_processes.append((context_name, process.exitcode))

        if failed_processes:
            raise RuntimeError(
                "Training subprocess failed: "
                + ", ".join(
                    f"{context_name} exitcode={exitcode}"
                    for context_name, exitcode in failed_processes
                )
            )


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else NTRAINING)