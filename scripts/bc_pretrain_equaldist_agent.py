#!/usr/bin/env python3
"""Behavior cloning from the real EqualDistOneShotAgent.

This script does NOT use scml.oneshot.rl.policies.greedy_policy.

Instead:
1. It runs a real EqualDistOneShotAgent subclass.
2. It records the exact observations seen by our MyObservationManager.
3. It records the real SAO responses returned by EqualDistOneShotAgent.
4. It encodes those responses with FlexibleActionManager.encode(...).
5. It trains the PPO policy by supervised learning.
6. It validates imitation quality before saving the warm-start model.

Output:
    myagent/models/bc_equaldist_true/mymodel<Context>.zip

Example:
    python scripts/bc_pretrain_equaldist_agent.py \
        --contexts BalancedSupplierContext \
        --n-samples 50000 \
        --bc-epochs 50 \
        --batch-size 512 \
        --seed 0
"""

from __future__ import annotations

import argparse
import copy
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from negmas import Outcome
from negmas.sao import SAOResponse, ResponseType
from scml.oneshot.agents import EqualDistOneShotAgent
from scml.oneshot.rl.action import FlexibleActionManager
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from myagent.common import ALL_CONTEXTS, LOG_ROOT, MODEL_PATH, TrainingAlgorithm, make_context
from myagent.train import MyObservationManager, evaluate_model, make_env


@dataclass
class DatasetStats:
    encoded_samples: int = 0
    encode_failures: int = 0
    invalid_encoded_actions: int = 0
    first_proposal_samples: int = 0
    counter_all_samples: int = 0
    worlds_run: int = 0


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value_float = float(value)
        if np.isfinite(value_float):
            return value_float
    except Exception:
        pass
    return default


def _agent_rank(scores: dict[str, float], selected_ids: list[str]) -> float:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ranks = {agent_id: rank for rank, (agent_id, _) in enumerate(ranked, start=1)}

    selected = [
        ranks[agent_id]
        for agent_id in selected_ids
        if agent_id in ranks
    ]

    if not selected:
        return float(len(scores))

    return float(min(selected))


def _run_world(world: Any) -> None:
    if hasattr(world, "run"):
        world.run()
    else:
        world.run_with_progress()


def _make_robust_context(context_name: str):
    context = make_context(context_name)
    context.world_params.update(
        dict(
            debug=False,
            ignore_agent_exceptions=True,
            ignore_negotiation_exceptions=True,
            ignore_contract_execution_exceptions=True,
            ignore_simulation_exceptions=True,
        )
    )
    return context


def _responses_from_first_proposals(
    proposals: dict[str, Outcome | None],
) -> dict[str, SAOResponse]:
    """Convert first_proposals() output to SAOResponse dict.

    In SCML, first_proposals returns offers. FlexibleActionManager.encode expects
    SAOResponse values. A non-None proposal is represented as REJECT_OFFER with
    an outcome, which is how counter-offers are represented later as well.
    """
    responses: dict[str, SAOResponse] = {}

    for partner, outcome in proposals.items():
        if outcome is None:
            responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
        else:
            responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, outcome)

    return responses


def _is_valid_action(action: Any, action_space: spaces.Space) -> bool:
    try:
        return bool(action_space.contains(action))
    except Exception:
        return False


def _as_action_array(action: Any, action_space: spaces.Space) -> np.ndarray:
    """Cast encoded action to stable dtype/shape without clipping."""
    if isinstance(action_space, spaces.Discrete):
        return np.asarray(int(action), dtype=np.int64)

    if isinstance(action_space, spaces.MultiDiscrete):
        return np.asarray(action, dtype=np.int64).reshape(action_space.shape)

    if isinstance(action_space, spaces.MultiBinary):
        return np.asarray(action, dtype=np.int64).reshape(action_space.shape)

    if isinstance(action_space, spaces.Box):
        return np.asarray(action, dtype=np.float32).reshape(action_space.shape)

    return np.asarray(action)


def _action_summary(actions: np.ndarray, action_space: spaces.Space) -> None:
    print("=== Action target summary ===")
    print(f"action shape: {actions.shape}")
    print(f"action dtype:  {actions.dtype}")

    if isinstance(action_space, spaces.MultiDiscrete):
        nvec = np.asarray(action_space.nvec).reshape(-1)
        flat = actions.reshape(actions.shape[0], -1)

        for idx in range(flat.shape[1]):
            values, counts = np.unique(flat[:, idx], return_counts=True)
            top = sorted(
                zip(values.tolist(), counts.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )[:10]
            formatted = ", ".join(f"{v}:{c}" for v, c in top)
            print(f"component {idx:02d} n={nvec[idx]} top: {formatted}")


def _score_summary(values: list[float]) -> str:
    if not values:
        return "n=0"

    arr = np.asarray(values, dtype=np.float32)
    return (
        f"n={len(values)} mean={float(np.mean(arr)):.6f} "
        f"std={float(np.std(arr)):.6f} "
        f"min={float(np.min(arr)):.6f} "
        f"max={float(np.max(arr)):.6f}"
    )


def make_recording_equaldist_agent_class(
    context_name: str,
    max_samples: int,
    observations: list[np.ndarray],
    actions: list[np.ndarray],
    stats: DatasetStats,
):
    """Create a subclass of EqualDistOneShotAgent that records obs/action pairs."""

    context = _make_robust_context(context_name)
    action_manager = FlexibleActionManager(context=context)
    observation_manager = MyObservationManager(context=context, continuous=True)
    action_space = action_manager.make_space()

    class RecordingEqualDistOneShotAgent(EqualDistOneShotAgent):
        """EqualDistOneShotAgent with BC data recording hooks."""

        def _record_responses(
            self,
            responses: dict[str, SAOResponse],
            source: str,
        ) -> None:
            if len(observations) >= max_samples:
                return

            try:
                obs = observation_manager.encode(self.awi)
                encoded = action_manager.encode(self.awi, responses)
                encoded = _as_action_array(encoded, action_space)

            except Exception as exc:
                stats.encode_failures += 1

                if stats.encode_failures <= 5 or stats.encode_failures % 1000 == 0:
                    print(
                        f"[encode failure] context={context_name} "
                        f"source={source} count={stats.encode_failures} error={exc}"
                    )
                return

            if not _is_valid_action(encoded, action_space):
                stats.invalid_encoded_actions += 1

                if (
                    stats.invalid_encoded_actions <= 5
                    or stats.invalid_encoded_actions % 1000 == 0
                ):
                    print(
                        f"[invalid encoded action] context={context_name} "
                        f"source={source} count={stats.invalid_encoded_actions} "
                        f"action={encoded.tolist()} action_space={action_space}"
                    )
                return

            observations.append(np.asarray(obs, dtype=np.float32).copy())
            actions.append(encoded.copy())
            stats.encoded_samples += 1

            if source == "first_proposals":
                stats.first_proposal_samples += 1
            elif source == "counter_all":
                stats.counter_all_samples += 1

        def first_proposals(self):
            proposals = super().first_proposals()
            responses = _responses_from_first_proposals(proposals)
            self._record_responses(responses, source="first_proposals")
            return proposals

        def counter_all(self, offers, states):
            responses = super().counter_all(offers, states)
            self._record_responses(responses, source="counter_all")
            return responses

    RecordingEqualDistOneShotAgent.__name__ = "RecordingEqualDistOneShotAgent"
    RecordingEqualDistOneShotAgent.__qualname__ = "RecordingEqualDistOneShotAgent"
    RecordingEqualDistOneShotAgent.__module__ = __name__

    globals()["RecordingEqualDistOneShotAgent"] = RecordingEqualDistOneShotAgent

    return RecordingEqualDistOneShotAgent, action_space


def collect_dataset(
    context_name: str,
    n_samples: int,
    seed: int,
    max_worlds: int,
) -> tuple[np.ndarray, np.ndarray, DatasetStats, spaces.Space]:
    """Collect obs/action pairs from the real EqualDistOneShotAgent."""
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    stats = DatasetStats()

    RecordingAgent, action_space = make_recording_equaldist_agent_class(
        context_name=context_name,
        max_samples=n_samples,
        observations=observations,
        actions=actions,
        stats=stats,
    )

    world_seed = seed
    consecutive_world_failures = 0

    while len(observations) < n_samples and stats.worlds_run < max_worlds:
        set_global_seed(world_seed)

        context = _make_robust_context(context_name)

        try:
            world, _agents = context.generate(
                types=(RecordingAgent,),
                params=(dict(),),
            )
            _run_world(world)
            consecutive_world_failures = 0

        except Exception as exc:
            print(
                f"[world failed] context={context_name} "
                f"world={stats.worlds_run} seed={world_seed} error={exc}"
            )

            consecutive_world_failures += 1

            if consecutive_world_failures >= 5 and len(observations) == 0:
                raise RuntimeError(
                    f"World generation failed {consecutive_world_failures} times "
                    f"in a row before collecting any sample. Last error: {exc}"
                ) from exc

        stats.worlds_run += 1
        world_seed += 1

        if stats.worlds_run % 10 == 0 or len(observations) >= n_samples:
            print(
                f"collection progress: context={context_name} "
                f"samples={len(observations)}/{n_samples} "
                f"worlds={stats.worlds_run} "
                f"encode_failures={stats.encode_failures} "
                f"invalid_encoded={stats.invalid_encoded_actions}"
            )

    if len(observations) < n_samples:
        raise RuntimeError(
            f"Could only collect {len(observations)} samples after "
            f"{stats.worlds_run} worlds. Requested {n_samples}."
        )

    obs_array = np.asarray(observations[:n_samples], dtype=np.float32)
    action_array = np.asarray(actions[:n_samples])

    return obs_array, action_array, stats, action_space


def make_ppo_model(context_name: str, run_name: str, seed: int | None):
    """Create PPO model with the same relevant architecture as myagent.train."""
    env = VecMonitor(DummyVecEnv([lambda: make_env(context_name)]))

    policy_kwargs = dict(
        net_arch=[128, 128],
    )

    model = TrainingAlgorithm(
        "MlpPolicy",
        env,
        verbose=0,
        policy_kwargs=policy_kwargs,
        seed=seed,
        tensorboard_log=f"./{LOG_ROOT}/tensorboard_logs/{run_name}/{context_name}",
    )

    return model, env


def split_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(observations)
    indices = np.random.default_rng(seed).permutation(n)

    n_val = int(round(n * validation_fraction))
    n_val = max(1, min(n - 1, n_val))

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return (
        observations[train_idx],
        actions[train_idx],
        observations[val_idx],
        actions[val_idx],
    )


def _evaluate_bc_loss(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
) -> float:
    device = model.device
    action_space = model.action_space
    losses: list[float] = []

    model.policy.set_training_mode(False)

    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            obs_batch = observations[start : start + batch_size]
            action_batch = actions[start : start + batch_size]

            obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)

            if isinstance(action_space, spaces.Box):
                action_tensor = torch.as_tensor(
                    action_batch,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                action_tensor = torch.as_tensor(
                    action_batch,
                    device=device,
                    dtype=torch.long,
                )

                if isinstance(action_space, spaces.Discrete):
                    action_tensor = action_tensor.reshape(-1)

            _values, log_prob, _entropy = model.policy.evaluate_actions(
                obs_tensor,
                action_tensor,
            )

            losses.append(float((-log_prob.mean()).detach().cpu()))

    model.policy.set_training_mode(True)

    return float(np.mean(losses)) if losses else float("nan")


def _predict_actions(
    model,
    observations: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    preds: list[np.ndarray] = []

    for start in range(0, len(observations), batch_size):
        obs_batch = observations[start : start + batch_size]
        action_batch, _ = model.predict(obs_batch, deterministic=True)
        preds.append(np.asarray(action_batch))

    return np.concatenate(preds, axis=0)


def _imitation_metrics(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    pred_actions = _predict_actions(model, observations, batch_size)

    pred_flat = pred_actions.reshape(pred_actions.shape[0], -1)
    target_flat = actions.reshape(actions.shape[0], -1)

    component_match = pred_flat == target_flat
    exact_match = np.all(component_match, axis=1)

    return {
        "component_accuracy": float(np.mean(component_match)),
        "exact_accuracy": float(np.mean(exact_match)),
    }


def behavior_clone(
    model,
    train_obs: np.ndarray,
    train_actions: np.ndarray,
    val_obs: np.ndarray,
    val_actions: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    ent_coef: float,
    max_grad_norm: float,
    patience: int,
    target_component_accuracy: float,
    target_exact_accuracy: float,
) -> dict[str, float]:
    """Supervised pretraining of PPO policy using expert obs/action pairs."""
    device = model.device
    action_space = model.action_space

    model.policy.set_training_mode(True)

    optimizer = model.policy.optimizer
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    best_state_dict = copy.deepcopy(model.policy.state_dict())
    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0

    n_samples = train_obs.shape[0]

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(n_samples)
        epoch_losses: list[float] = []
        epoch_entropies: list[float] = []

        model.policy.set_training_mode(True)

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            obs_batch = train_obs[batch_idx]
            action_batch = train_actions[batch_idx]

            obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)

            if isinstance(action_space, spaces.Box):
                action_tensor = torch.as_tensor(
                    action_batch,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                action_tensor = torch.as_tensor(
                    action_batch,
                    device=device,
                    dtype=torch.long,
                )

                if isinstance(action_space, spaces.Discrete):
                    action_tensor = action_tensor.reshape(-1)

            _values, log_prob, entropy = model.policy.evaluate_actions(
                obs_tensor,
                action_tensor,
            )

            loss = -log_prob.mean()

            if entropy is not None and ent_coef != 0.0:
                loss = loss - ent_coef * entropy.mean()
                epoch_entropies.append(float(entropy.mean().detach().cpu()))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_grad_norm)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = _evaluate_bc_loss(model, val_obs, val_actions, batch_size)
        val_metrics = _imitation_metrics(model, val_obs, val_actions, batch_size)

        current_metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.policy.state_dict())
            best_metrics = current_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        mean_entropy = (
            float(np.mean(epoch_entropies)) if epoch_entropies else float("nan")
        )

        print(
            f"epoch={epoch:03d}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_component_acc={val_metrics['component_accuracy']:.4f} "
            f"val_exact_acc={val_metrics['exact_accuracy']:.4f} "
            f"entropy={mean_entropy:.6f} "
            f"best_val_loss={best_val_loss:.6f}"
        )

        if (
            val_metrics["component_accuracy"] >= target_component_accuracy
            and val_metrics["exact_accuracy"] >= target_exact_accuracy
        ):
            print(
                "Reached target imitation quality: "
                f"component_accuracy={val_metrics['component_accuracy']:.4f}, "
                f"exact_accuracy={val_metrics['exact_accuracy']:.4f}"
            )
            break

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without validation improvement.")
            break

    model.policy.load_state_dict(best_state_dict)
    return best_metrics


def evaluate_bc_model(
    model,
    context_name: str,
    n_episodes: int,
) -> dict[str, float]:
    my_scores = []
    score_gaps = []
    gap_vs_best = []
    ranks = []

    for seed in range(n_episodes):
        metrics = evaluate_model(model, context_name, seed=seed)

        if "my_score" in metrics:
            my_scores.append(float(metrics["my_score"]))
        if "score_gap" in metrics:
            score_gaps.append(float(metrics["score_gap"]))
        if "score_gap_vs_best_opponent" in metrics:
            gap_vs_best.append(float(metrics["score_gap_vs_best_opponent"]))
        if "my_rank" in metrics:
            ranks.append(float(metrics["my_rank"]))

    return {
        "my_score_mean": float(np.mean(my_scores)) if my_scores else float("nan"),
        "score_gap_mean": float(np.mean(score_gaps)) if score_gaps else float("nan"),
        "gap_vs_best_mean": float(np.mean(gap_vs_best)) if gap_vs_best else float("nan"),
        "rank_mean": float(np.mean(ranks)) if ranks else float("nan"),
    }


def evaluate_true_equaldist(
    context_name: str,
    n_episodes: int,
) -> dict[str, float]:
    my_scores = []
    ranks = []

    for seed in range(n_episodes):
        set_global_seed(seed)
        context = _make_robust_context(context_name)

        world, agents = context.generate(
            types=(EqualDistOneShotAgent,),
            params=(dict(),),
        )

        _run_world(world)

        raw_scores = world.scores()
        scores = {
            str(agent_id): float(score)
            for agent_id, score in raw_scores.items()
            if np.isfinite(float(score))
        }

        agent_ids = [
            getattr(agent, "id", None)
            for agent in agents
            if getattr(agent, "id", None) is not None
        ]
        agent_ids = [str(agent_id) for agent_id in agent_ids]

        selected_scores = [
            scores[agent_id]
            for agent_id in agent_ids
            if agent_id in scores
        ]

        if selected_scores:
            my_scores.append(float(np.mean(selected_scores)))
            ranks.append(_agent_rank(scores, agent_ids))

    return {
        "my_score_mean": float(np.mean(my_scores)) if my_scores else float("nan"),
        "rank_mean": float(np.mean(ranks)) if ranks else float("nan"),
    }


def parse_contexts(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(ALL_CONTEXTS)

    return [
        item.strip()
        for item in raw.split(",")
        if item.strip()
    ]


def quality_passed(
    imitation_metrics: dict[str, float],
    bc_rollout: dict[str, float],
    expert_rollout: dict[str, float],
    min_component_accuracy: float,
    min_exact_accuracy: float,
    min_score_ratio: float,
) -> bool:
    component_ok = (
        imitation_metrics.get("component_accuracy", 0.0) >= min_component_accuracy
    )
    exact_ok = imitation_metrics.get("exact_accuracy", 0.0) >= min_exact_accuracy

    expert_score = expert_rollout.get("my_score_mean", float("nan"))
    bc_score = bc_rollout.get("my_score_mean", float("nan"))

    if min_score_ratio <= 0.0:
        score_ok = True
    elif np.isfinite(expert_score) and expert_score > 0.0 and np.isfinite(bc_score):
        score_ok = bc_score >= min_score_ratio * expert_score
    else:
        score_ok = False

    return component_ok and exact_ok and score_ok


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--contexts",
        default=os.environ.get("TRAIN_CONTEXTS", "BalancedSupplierContext"),
        help="Comma-separated context list or 'all'.",
    )
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--max-worlds", type=int, default=5000)
    parser.add_argument("--bc-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--target-component-accuracy", type=float, default=0.95)
    parser.add_argument("--target-exact-accuracy", type=float, default=0.70)
    parser.add_argument("--min-component-accuracy", type=float, default=0.90)
    parser.add_argument("--min-exact-accuracy", type=float, default=0.50)
    parser.add_argument("--min-score-ratio", type=float, default=0.85)

    parser.add_argument("--rollout-eval-episodes", type=int, default=5)
    parser.add_argument(
        "--save-even-if-bad",
        action="store_true",
        help="Save the model even if imitation or rollout quality is bad.",
    )
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help="Also save obs/actions as compressed .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save BC-pretrained models. Default: myagent/models/bc_equaldist_true",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="TensorBoard run name. Default: bc_equaldist_true",
    )

    args = parser.parse_args()

    os.environ.setdefault("LOG_REWARD_COMPONENTS", "0")
    set_global_seed(args.seed)

    run_name = args.run_name or "bc_equaldist_true"
    output_dir = Path(args.output_dir or (MODEL_PATH.parent / "bc_equaldist_true"))
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts = parse_contexts(args.contexts)

    print("=== True EqualDist behavior cloning ===")
    print(f"contexts: {contexts}")
    print(f"n_samples per context: {args.n_samples}")
    print(f"bc_epochs: {args.bc_epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"validation_fraction: {args.validation_fraction}")
    print(f"output_dir: {output_dir}")
    print(f"run_name: {run_name}")

    for context_name in contexts:
        print(f"\n=== Context: {context_name} ===")

        observations, actions, dataset_stats, action_space = collect_dataset(
            context_name=context_name,
            n_samples=args.n_samples,
            seed=args.seed,
            max_worlds=args.max_worlds,
        )

        print("=== Dataset collection stats ===")
        print(dataset_stats)
        print(f"observations: {observations.shape}")
        print(f"actions:      {actions.shape}")
        _action_summary(actions, action_space)

        if dataset_stats.encode_failures > 0 or dataset_stats.invalid_encoded_actions > 0:
            raise RuntimeError(
                "Dataset contains encoding failures or invalid encoded actions. "
                "Refusing to train a clone on damaged labels."
            )

        if args.save_dataset:
            dataset_path = output_dir / f"dataset_{context_name}_equaldist_true.npz"
            np.savez_compressed(
                dataset_path,
                observations=observations,
                actions=actions,
            )
            print(f"saved dataset: {dataset_path}")

        train_obs, train_actions, val_obs, val_actions = split_dataset(
            observations=observations,
            actions=actions,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )

        print("=== Dataset split ===")
        print(f"train_obs: {train_obs.shape}")
        print(f"train_actions: {train_actions.shape}")
        print(f"val_obs: {val_obs.shape}")
        print(f"val_actions: {val_actions.shape}")

        model, env = make_ppo_model(
            context_name=context_name,
            run_name=run_name,
            seed=args.seed,
        )

        try:
            best_metrics = behavior_clone(
                model=model,
                train_obs=train_obs,
                train_actions=train_actions,
                val_obs=val_obs,
                val_actions=val_actions,
                epochs=args.bc_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                ent_coef=args.ent_coef,
                max_grad_norm=args.max_grad_norm,
                patience=args.patience,
                target_component_accuracy=args.target_component_accuracy,
                target_exact_accuracy=args.target_exact_accuracy,
            )

            final_val_metrics = _imitation_metrics(
                model=model,
                observations=val_obs,
                actions=val_actions,
                batch_size=args.batch_size,
            )

            print("=== Best validation metrics ===")
            print(best_metrics)

            print("=== Final loaded-best imitation metrics ===")
            print(final_val_metrics)

            if args.rollout_eval_episodes > 0:
                print("=== Rollout evaluation: BC model ===")
                bc_rollout = evaluate_bc_model(
                    model=model,
                    context_name=context_name,
                    n_episodes=args.rollout_eval_episodes,
                )
                print(bc_rollout)

                print("=== Rollout evaluation: true EqualDistOneShotAgent ===")
                expert_rollout = evaluate_true_equaldist(
                    context_name=context_name,
                    n_episodes=args.rollout_eval_episodes,
                )
                print(expert_rollout)
            else:
                bc_rollout = {
                    "my_score_mean": float("nan"),
                    "rank_mean": float("nan"),
                }
                expert_rollout = {
                    "my_score_mean": float("nan"),
                    "rank_mean": float("nan"),
                }

            passed = quality_passed(
                imitation_metrics=final_val_metrics,
                bc_rollout=bc_rollout,
                expert_rollout=expert_rollout,
                min_component_accuracy=args.min_component_accuracy,
                min_exact_accuracy=args.min_exact_accuracy,
                min_score_ratio=args.min_score_ratio,
            )

            print(f"quality_passed: {passed}")

            if not passed and not args.save_even_if_bad:
                raise RuntimeError(
                    "Clone quality check failed. Not saving model. "
                    "Use --save-even-if-bad only for debugging."
                )

            save_path = output_dir / f"{MODEL_PATH.name}{context_name}"
            model.save(save_path)
            print(f"saved BC model: {save_path}.zip")

        finally:
            env.close()


if __name__ == "__main__":
    main()