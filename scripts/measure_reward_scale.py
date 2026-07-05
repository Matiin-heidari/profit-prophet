"""Measure the empirical scale of the reward terms per context.

This is a *diagnostic*, not a training run. It steps each context's env with
random actions for a few thousand steps and reports the distribution of the
(unweighted) score delta alongside the mean magnitude of every post-weight
reward term. The goal is to pick `score_delta_weight` / shaping scales from data
before launching a real training run.

Note: random actions rarely close deals, so the `deal_bonus` / `need_penalty`
columns reflect the "mostly no-deal" regime. The number this probe pins down is
the *score_delta* distribution, which is policy-independent in magnitude and is
the missing piece for setting the weights. The shaping maxima are known
analytically (need <= 0.40, deal/engagement 0.10, price ~ +/-0.02).

Usage:
    .venv/bin/python scripts/measure_reward_scale.py [n_steps_per_context]
"""

import glob
import os
import sys

# Allow running as `python scripts/measure_reward_scale.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must be set before importing the reward function so it logs components and
# does not collide with real runs.
os.environ["LOG_REWARD_COMPONENTS"] = "1"
os.environ.setdefault("RUN_NAME", "reward_scale_probe")
os.environ.setdefault("LOG_WORLD", "0")  # robust world (ignore exceptions)

import numpy as np
import pandas as pd

from myagent.common import ALL_CONTEXTS, LOG_ROOT
from myagent.train import make_env

# Post-weight reward terms logged by MyRewardFunction (all already scaled).
TERM_COLUMNS = [
    "score_delta_bonus",
    "need_penalty",
    "shortfall_penalty_term",
    "overshoot_penalty",
    "disposal_penalty_term",
    "productivity_bonus",
    "price_bonus",
    "deal_bonus",
    "engagement_bonus",
    "final_reward",
]


def _log_dir_for(context_name: str) -> str:
    run_name = os.environ.get("RUN_NAME", "default")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    return os.path.join(LOG_ROOT, "reward_component_logs", run_name, context_name, job_id)


def _roll_out(context_name: str, n_steps: int) -> str:
    """Step a context's env with random actions; return its log directory."""
    log_dir = _log_dir_for(context_name)
    # Start clean so we only read this probe's rows.
    for old in glob.glob(os.path.join(log_dir, "*.csv")):
        os.remove(old)

    env = make_env(context_name)
    env.reset(seed=0)

    steps = 0
    while steps < n_steps:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            env.reset()

    # Flush the per-step CSV.
    env._reward_function._close_reward_log()  # type: ignore[attr-defined]
    env.close()
    return log_dir


def _load_rows(log_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(log_dir, "*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def _report(context_name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {context_name}  (n={len(df)} steps) ===")
    if df.empty:
        print("  no rows logged")
        return

    sd = df["score_delta"].to_numpy(dtype=float)
    sd = sd[np.isfinite(sd)]
    if sd.size:
        print(
            "  score_delta (raw, unweighted): "
            f"mean={sd.mean():+.5f}  std={sd.std():.5f}  "
            f"p10={np.percentile(sd, 10):+.5f}  "
            f"p50={np.percentile(sd, 50):+.5f}  "
            f"p90={np.percentile(sd, 90):+.5f}  "
            f"mean|.|={np.abs(sd).mean():.5f}"
        )

    # Mean magnitude of each post-weight term (apples-to-apples dominance).
    rows = []
    for col in TERM_COLUMNS:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            continue
        rows.append((col, np.abs(vals).mean(), vals.mean(), vals.std()))

    if rows:
        # Sort by mean magnitude so the dominant term is first.
        rows.sort(key=lambda r: r[1], reverse=True)
        print("  post-weight term magnitudes (mean|.|, mean, std):")
        for name, mag, mean, std in rows:
            print(f"    {name:24s} mean|.|={mag:.5f}  mean={mean:+.5f}  std={std:.5f}")

    # The headline ratio: profit term vs the largest shaping term.
    shaping = [r for r in rows if r[0] not in ("score_delta_bonus", "final_reward")]
    sdb = next((r for r in rows if r[0] == "score_delta_bonus"), None)
    if shaping and sdb is not None:
        top = max(shaping, key=lambda r: r[1])
        ratio = sdb[1] / top[1] if top[1] > 0 else float("inf")
        print(
            f"  -> score_delta_bonus / largest shaping ({top[0]}) = {ratio:.2f}x "
            f"({'profit dominates' if ratio >= 1 else 'shaping dominates'})"
        )


def main() -> None:
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500

    print(f"Measuring reward scale: {n_steps} steps/context, "
          f"contexts={ALL_CONTEXTS}")

    for context_name in ALL_CONTEXTS:
        log_dir = _roll_out(context_name, n_steps)
        df = _load_rows(log_dir)
        _report(context_name, df)


if __name__ == "__main__":
    main()
