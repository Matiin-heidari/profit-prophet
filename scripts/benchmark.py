"""Benchmark MyAgent against the newest available ANAC OneShot qualifiers.

This is the measurement step: it drops MyAgent into a real ANAC-style tournament
against the qualifier pool (a meaningful opponent set with a real ceiling, unlike
the weak default agents) and reports where MyAgent lands. Run it over enough
configs that the ranking is not dominated by world-draw noise.

By default it uses the newest agent pool shipped by scml-agents (2025 as of
scml-agents 0.5.0 — note the *world* is still anac2024_oneshot; only the agent
pool is newer). Pass --year to pin a specific pool.

MyAgent loads its per-context models from MODEL_PATH on construction (see
myagent/myagent.py), so the benchmark just needs to pass the class in as a
competitor — the tournament framework instantiates it.

Usage:
    .venv/bin/python scripts/benchmark.py --n-configs 10 --n-steps 50
    .venv/bin/python scripts/benchmark.py --year 2024 --include-defaults
"""

import argparse
import datetime
import glob
import os
import random
import sys
import time

# Allow running as `python scripts/benchmark.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
from negmas.helpers import humanize_time
from scml.utils import anac2024_oneshot, DefaultAgentsOneShot2024
from scml_agents import get_agents

from myagent.common import LOG_ROOT
from myagent.myagent import MyAgent


def _short(name: str) -> str:
    return str(name).split(".")[-1]


def newest_agent_year() -> int:
    """Return the newest year for which scml-agents ships a OneShot pool.

    scml-agents raises ValueError instantly for unknown years (no import cost),
    so we can just probe downward from a year safely past the present. This keeps
    the benchmark on the latest pool automatically across future scml-agents
    upgrades instead of hard-coding a year.
    """
    for year in range(datetime.date.today().year + 1, 2018, -1):
        try:
            if get_agents(year, track="oneshot", as_class=True, ignore_failing=True):
                return year
        except ValueError:
            continue
    return 2024  # fallback: the last pool we know exists


def load_qualifiers(year: int) -> tuple[list, bool]:
    """Load the OneShot competitor pool for ``year``, preferring qualified-only.

    Some pools ship broken qualification metadata (e.g. 2025 in scml-agents
    0.5.0, whose qualified_only path raises AttributeError on a malformed agent
    module). In that case we fall back to the full pool so the benchmark still
    runs against the newest agents. Returns (agents, used_qualified_filter).
    """
    try:
        agents = list(
            get_agents(
                year,
                track="oneshot",
                qualified_only=True,
                as_class=True,
                ignore_failing=True,  # skip qualifiers that no longer import
            )
        )
        if agents:
            return agents, True
    except Exception as e:
        print(
            f"(qualified-only pool for {year} unavailable "
            f"[{type(e).__name__}: {e}] — falling back to the full pool)"
        )
    agents = list(
        get_agents(
            year,
            track="oneshot",
            qualified_only=False,
            as_class=True,
            ignore_failing=True,
        )
    )
    return agents, False


def report_context_usage(run: str | None = None) -> None:
    """Aggregate MyAgent's context-usage logs and print which per-context model
    (or the Greedy fallback) it actually selected, as a share of all worlds.

    MyAgent writes one row per world to log/context_usage_logs/<RUN_NAME>/usage_*.csv
    (see MyAgent._log_context_usage). This tells us not just how MyAgent scored
    but whether routing is healthy — e.g. an over-selection of one index-0
    context, or a high fallback rate, both of which mean the models aren't
    really being used as intended.
    """
    run = run or os.environ.get("RUN_NAME", "default")
    log_dir = os.path.join(LOG_ROOT, "context_usage_logs", run)
    files = sorted(glob.glob(os.path.join(log_dir, "usage_*.csv")))
    if not files:
        print(f"\n(no context-usage logs found under {log_dir}/ — "
              f"set RUN_NAME and LOG_CONTEXT_USAGE=1 to enable)")
        return

    try:
        frames = [pd.read_csv(f) for f in files]
        usage = pd.concat(frames, ignore_index=True)
    except Exception as e:
        print(f"\n(context-usage aggregation unavailable: {e})")
        return

    total = len(usage)
    if total == 0:
        print(f"\n(context-usage logs under {log_dir}/ are empty)")
        return

    counts = usage["chosen"].value_counts()
    fallback = int(counts.get("fallback", 0))
    print(f"\n=== Context usage ({total} worlds, {len(files)} log files, "
          f"run='{run}') ===")
    for name, n in counts.items():
        if name == "fallback":
            continue
        print(f"  {name:<28} {n:>6}  {100.0 * n / total:5.1f}%")
    print(f"  {'-' * 28} {'-' * 6}")
    print(f"  {'fallback (Greedy)':<28} {fallback:>6}  "
          f"{100.0 * fallback / total:5.1f}%")


def report_ranking(scores: pd.DataFrame) -> None:
    """Rank agents by mean score from a per-agent-per-world scores frame.

    Expects columns `agent_type` (already shortened) and `score`, one row per
    agent per world. Shared by benchmark() and scripts/aggregate_benchmark.py so a
    sharded run reports identically to a single run.
    """
    agg = (
        scores.groupby("agent_type")["score"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    agg.index = agg.index + 1  # 1-based rank
    print("\n=== Tournament ranking (mean score / std / n, best first) ===")
    print(agg.to_string())

    mine = agg[agg["agent_type"].str.contains("MyAgent")]
    if not mine.empty:
        rank = int(mine.index[0])
        my_score = float(mine.iloc[0]["mean"])
        best_score = float(agg.iloc[0]["mean"])
        print(
            f"\nMyAgent: rank {rank} / {len(agg)}   "
            f"score={my_score:.4f}   best={best_score:.4f}   "
            f"gap_to_best={my_score - best_score:+.4f}   "
            f"(worlds={int(mine.iloc[0]['count'])})"
        )
    else:
        print("\nMyAgent not found in results (did it fail to instantiate?).")


def benchmark(
    year: int | None = None,
    n_configs: int = 10,
    n_steps: int = 50,
    include_defaults: bool = False,
    serial: bool = False,
    save_scores: str | None = None,
) -> None:
    if year is None:
        year = newest_agent_year()
    qualifiers, used_qualified = load_qualifiers(year)
    pool_kind = "qualifier" if used_qualified else "full-pool"
    print(f"Loaded {len(qualifiers)} {pool_kind} agents for {year}.")

    # scml_agents has a module-level `random.seed(0)` (scml2022 oneshot team_131,
    # imported transitively by get_agents/load_qualifiers above) that silently
    # fixes Python's global random/np.random state on import. Left alone, every
    # fresh process (e.g. every sharded SLURM task) draws the identical "first"
    # sample from anac2024_oneshot's config generator (same n_steps, same
    # per-level population split) instead of a genuinely different world.
    # Reseed from OS entropy here, after the pollution and before the
    # tournament call consumes it, to restore real per-run randomness.
    random.seed()
    np.random.seed()

    competitors = [MyAgent] + qualifiers
    if include_defaults:
        competitors += list(DefaultAgentsOneShot2024)

    print(
        f"Running tournament: {len(competitors)} competitors, "
        f"n_configs={n_configs}, n_steps={n_steps} "
        f"({'serial' if serial else 'parallel'})"
    )

    start = time.perf_counter()
    results = anac2024_oneshot(
        competitors=competitors,
        verbose=True,
        n_steps=n_steps,
        n_configs=n_configs,
        debug=False,
        parallelism="serial" if serial else "parallel",
        # Don't reveal type/position in names — keeps the comparison honest.
        agent_name_reveals_position=False,
        agent_name_reveals_type=False,
    )
    elapsed = time.perf_counter() - start

    # --- Reporting -----------------------------------------------------------
    # results.scores is the granular per-agent-per-world frame; the per-agent mean
    # is the ranking. Working from it (not total_scores) means a sharded run can
    # concatenate these frames and re-rank identically.
    scores = results.scores.copy()  # type: ignore
    scores["agent_type"] = scores["agent_type"].map(_short)

    if save_scores:
        os.makedirs(os.path.dirname(save_scores) or ".", exist_ok=True)
        scores[["agent_type", "score"]].to_csv(save_scores, index=False)
        print(f"Saved {len(scores)} agent-world scores -> {save_scores}")

    report_ranking(scores)

    # Which per-context model (or fallback) MyAgent actually used, from its
    # per-world usage logs. Surfaces routing problems the score alone can't.
    report_context_usage()

    print(f"\nFinished in {humanize_time(elapsed)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=None,
                   help="agent-pool year (default: newest available)")
    p.add_argument("--n-configs", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--include-defaults", action="store_true",
                   help="also include the weak DefaultAgentsOneShot2024 pool")
    p.add_argument("--serial", action="store_true")
    p.add_argument("--save-scores", default=None,
                   help="write per-agent-per-world scores to this CSV (for "
                        "sharded runs; aggregate with scripts/aggregate_benchmark.py)")
    args = p.parse_args()
    benchmark(
        year=args.year,
        n_configs=args.n_configs,
        n_steps=args.n_steps,
        include_defaults=args.include_defaults,
        serial=args.serial,
        save_scores=args.save_scores,
    )


if __name__ == "__main__":
    main()
