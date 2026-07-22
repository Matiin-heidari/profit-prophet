#!/usr/bin/env python3
"""Behavior-cloning pretraining for SCML OneShot PPO policies.

This script creates a PPO model with the same policy architecture as myagent.train,
collects observation/action pairs from a simple rule-based expert policy, and trains
the PPO policy by supervised learning on those expert actions.

Default expert:
    equal
        Uses SCML's greedy_policy with an equal distributor. This is not a
        byte-for-byte clone of EqualDistOneShotAgent, but it gives us an
        EqualDist-style expert in the exact FlexibleActionManager action space.

Alternative:
    greedy
        Uses SCML's default greedy_policy.

Output:
    myagent/models/bc_<expert>/mymodel<Context>.zip

Example:
    python scripts/bc_pretrain_expert.py \
        --expert equal \
        --contexts BalancedSupplierContext \
        --n-samples 50000 \
        --bc-epochs 8 \
        --batch-size 512 \
        --seed 0
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from myagent.common import CONTEXTS, LOG_ROOT, MODEL_PATH, TrainingAlgorithm
from myagent.train import make_env


def _import_greedy_policy() -> Callable:
    """Import SCML's encoded RL greedy policy.

    SCML versions have moved pieces around, so we try the oneshot location first
    and then the std location documented in the current docs.
    """
    errors: list[str] = []

    for module_name in (
        "scml.oneshot.rl.policies",
        "scml.std.rl.policies",
    ):
        try:
            module = __import__(module_name, fromlist=["greedy_policy"])
            return getattr(module, "greedy_policy")
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")

    raise ImportError(
        "Could not import greedy_policy from SCML. Tried:\n"
        + "\n".join(errors)
    )


def equal_distributor(quantity: int, n_partners: int) -> list[int]:
    """Distribute a non-negative integer quantity as evenly as possible."""
    if n_partners <= 0:
        return []

    quantity = max(0, int(quantity))
    base = quantity // n_partners
    remainder = quantity % n_partners

    return [
        base + (1 if idx < remainder else 0)
        for idx in range(n_partners)
    ]


def _get_nested_attr(obj: Any, path: str) -> Any | None:
    current = obj

    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)

    return current


def _first_existing_attr(obj: Any, candidates: tuple[str, ...]) -> Any | None:
    for candidate in candidates:
        value = _get_nested_attr(obj, candidate)
        if value is not None:
            return value

    return None


def resolve_env_parts(env: Any) -> tuple[Any, Any, Any]:
    """Find AWI, observation manager and action manager on a OneShotEnv."""
    awi = _first_existing_attr(
        env,
        (
            "awi",
            "_awi",
            "agent.awi",
            "_agent.awi",
            "_rl_agent.awi",
            "rl_agent.awi",
        ),
    )

    obs_manager = _first_existing_attr(
        env,
        (
            "observation_manager",
            "_observation_manager",
            "obs_manager",
            "_obs_manager",
        ),
    )

    action_manager = _first_existing_attr(
        env,
        (
            "action_manager",
            "_action_manager",
        ),
    )

    missing = []
    if awi is None:
        missing.append("awi")
    if obs_manager is None:
        missing.append("observation_manager")
    if action_manager is None:
        missing.append("action_manager")

    if missing:
        raise RuntimeError(
            "Could not resolve required OneShotEnv attributes: "
            + ", ".join(missing)
            + "\nAvailable env attrs include:\n"
            + ", ".join(sorted(name for name in dir(env) if not name.startswith("__"))[:200])
        )

    return awi, obs_manager, action_manager


def reset_env(env: Any, seed: int | None = None) -> np.ndarray:
    """Reset Gym/Gymnasium env and return obs."""
    try:
        result = env.reset(seed=seed)
    except TypeError:
        result = env.reset()

    if isinstance(result, tuple):
        obs, _info = result
    else:
        obs = result

    return np.asarray(obs, dtype=np.float32)


def step_env(env: Any, action: Any) -> tuple[np.ndarray, bool]:
    """Step Gym/Gymnasium env and return (obs, done)."""
    result = env.step(action)

    if len(result) == 5:
        obs, _reward, terminated, truncated, _info = result
        done = bool(terminated or truncated)
    elif len(result) == 4:
        obs, _reward, done, _info = result
        done = bool(done)
    else:
        raise RuntimeError(f"Unexpected env.step result length: {len(result)}")

    return np.asarray(obs, dtype=np.float32), done


def sanitize_action(action: Any, action_space: spaces.Space) -> Any:
    """Cast expert action into the dtype/shape expected by the env.

    For MultiDiscrete spaces, SCML's helper policies can sometimes emit negative
    components for one side of the market. The environment cannot consume these,
    so we clip every component into the legal [0, n_i - 1] range. This gives us a
    valid BC target instead of aborting the whole data collection.
    """
    if isinstance(action_space, spaces.Discrete):
        raw = int(np.asarray(action).reshape(-1)[0])
        return int(np.clip(raw, 0, action_space.n - 1))

    if isinstance(action_space, spaces.MultiDiscrete):
        raw_array = np.asarray(action, dtype=np.int64).reshape(action_space.shape)
        low = np.zeros_like(action_space.nvec, dtype=np.int64).reshape(action_space.shape)
        high = (np.asarray(action_space.nvec, dtype=np.int64) - 1).reshape(action_space.shape)
        repaired = np.clip(raw_array, low, high).astype(np.int64)

        if not np.array_equal(raw_array, repaired):
            count = getattr(sanitize_action, "_repair_count", 0) + 1
            setattr(sanitize_action, "_repair_count", count)

            if count <= 5 or count % 10000 == 0:
                print(
                    "[action repair] clipped invalid MultiDiscrete expert action "
                    f"#{count}: raw={raw_array.tolist()} repaired={repaired.tolist()}"
                )

        return repaired

    if isinstance(action_space, spaces.MultiBinary):
        raw_array = np.asarray(action, dtype=np.int64).reshape(action_space.shape)
        return np.clip(raw_array, 0, 1).astype(np.int64)

    if isinstance(action_space, spaces.Box):
        action_array = np.asarray(action, dtype=np.float32)
        return np.clip(action_array, action_space.low, action_space.high)

    return action

def expert_action(
    expert: str,
    greedy_policy: Callable,
    obs: np.ndarray,
    env: Any,
) -> Any:
    """Generate an encoded action using SCML's RL policy function."""
    awi, obs_manager, action_manager = resolve_env_parts(env)

    kwargs: dict[str, Any] = {
        "obs": obs,
        "awi": awi,
        "obs_manager": obs_manager,
        "action_manager": action_manager,
        "debug": False,
    }

    signature = inspect.signature(greedy_policy)

    if expert in ("equal", "equaldist", "equaldistoneshot"):
        if "distributor" in signature.parameters:
            kwargs["distributor"] = equal_distributor
        else:
            raise RuntimeError(
                "Selected expert='equal', but this SCML greedy_policy does not "
                "accept a distributor argument."
            )

    try:
        action = greedy_policy(**kwargs)
    except TypeError:
        # Fallback for positional signatures.
        if expert in ("equal", "equaldist", "equaldistoneshot"):
            action = greedy_policy(
                obs,
                awi,
                obs_manager,
                action_manager,
                False,
                equal_distributor,
            )
        else:
            action = greedy_policy(obs, awi, obs_manager, action_manager)

    return sanitize_action(action, env.action_space)


def collect_dataset(
    context_name: str,
    expert: str,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect observation/action pairs from the selected expert policy."""
    greedy_policy = _import_greedy_policy()
    env = make_env(context_name)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    try:
        episode = 0
        obs = reset_env(env, seed=seed)

        while len(observations) < n_samples:
            action = expert_action(expert, greedy_policy, obs, env)

            if not env.action_space.contains(action):
                raise RuntimeError(
                    "Expert produced invalid action.\n"
                    f"context={context_name}\n"
                    f"expert={expert}\n"
                    f"action={action}\n"
                    f"action_space={env.action_space}"
                )

            observations.append(np.asarray(obs, dtype=np.float32).copy())
            actions.append(np.asarray(action).copy())

            obs, done = step_env(env, action)

            if done:
                episode += 1
                obs = reset_env(env, seed=seed + episode)

    finally:
        env.close()

    obs_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions)

    return obs_array, action_array


def make_ppo_model(context_name: str, run_name: str, seed: int | None):
    """Create PPO model with same relevant architecture as myagent.train."""
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


def behavior_clone(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    ent_coef: float,
    max_grad_norm: float,
) -> None:
    """Supervised pretraining of PPO policy using expert obs/action pairs."""
    device = model.device
    action_space = model.action_space

    model.policy.set_training_mode(True)

    # Use the policy optimizer but set a BC-specific learning rate.
    optimizer = model.policy.optimizer
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    n_samples = observations.shape[0]

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(n_samples)
        epoch_losses: list[float] = []
        epoch_entropies: list[float] = []

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            obs_batch = observations[batch_idx]
            action_batch = actions[batch_idx]

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

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        mean_entropy = (
            float(np.mean(epoch_entropies)) if epoch_entropies else float("nan")
        )

        print(
            f"epoch={epoch:03d}/{epochs} "
            f"bc_loss={mean_loss:.6f} "
            f"entropy={mean_entropy:.6f}"
        )


def parse_contexts(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(CONTEXTS)

    return [
        item.strip()
        for item in raw.split(",")
        if item.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expert",
        choices=["equal", "equaldist", "equaldistoneshot", "greedy"],
        default="equal",
        help="Expert policy to clone. 'equal' uses greedy_policy with equal distributor.",
    )
    parser.add_argument(
        "--contexts",
        default=os.environ.get("TRAIN_CONTEXTS", "BalancedSupplierContext"),
        help="Comma-separated context list or 'all'.",
    )
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save BC-pretrained models. Default: myagent/models/bc_<expert>",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="TensorBoard run name. Default: bc_<expert>",
    )
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help="Also save obs/actions as compressed .npz files.",
    )

    args = parser.parse_args()

    # Reward-component logs would be huge and are not useful for BC collection.
    os.environ.setdefault("LOG_REWARD_COMPONENTS", "0")

    expert_name = "equal" if args.expert in ("equaldist", "equaldistoneshot") else args.expert
    run_name = args.run_name or f"bc_{expert_name}"
    output_dir = Path(args.output_dir or (MODEL_PATH.parent / f"bc_{expert_name}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts = parse_contexts(args.contexts)

    print("=== Behavior cloning pretraining ===")
    print(f"expert: {args.expert}")
    print(f"contexts: {contexts}")
    print(f"n_samples per context: {args.n_samples}")
    print(f"bc_epochs: {args.bc_epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"output_dir: {output_dir}")
    print(f"run_name: {run_name}")

    for context_name in contexts:
        print(f"\n=== Context: {context_name} ===")

        observations, actions = collect_dataset(
            context_name=context_name,
            expert=args.expert,
            n_samples=args.n_samples,
            seed=args.seed,
        )

        print(f"dataset observations: {observations.shape}")
        print(f"dataset actions:      {actions.shape}")

        if args.save_dataset:
            dataset_path = output_dir / f"dataset_{context_name}_{expert_name}.npz"
            np.savez_compressed(
                dataset_path,
                observations=observations,
                actions=actions,
            )
            print(f"saved dataset: {dataset_path}")

        model, env = make_ppo_model(
            context_name=context_name,
            run_name=run_name,
            seed=args.seed,
        )

        try:
            behavior_clone(
                model=model,
                observations=observations,
                actions=actions,
                epochs=args.bc_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                ent_coef=args.ent_coef,
                max_grad_norm=args.max_grad_norm,
            )

            save_path = output_dir / f"{MODEL_PATH.name}{context_name}"
            model.save(save_path)
            print(f"saved BC model: {save_path}.zip")

        finally:
            env.close()


if __name__ == "__main__":
    main()