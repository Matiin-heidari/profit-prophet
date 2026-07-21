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
import hashlib
import os
import random
import shutil
import sys
import time
from math import comb

# Allow running as `python scripts/benchmark.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# --model-dir must take effect BEFORE `from myagent...` below: MODEL_PATH is
# resolved from the MODEL_DIR env var at import time (myagent/common.py). This
# pre-scan sets the env var (which forked tournament workers also inherit); the
# real parser in main() re-declares the flag so it shows up in --help.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--model-dir", default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.model_dir:
    os.environ["MODEL_DIR"] = _pre_args.model_dir

import numpy as np
import pandas as pd
import negmas.tournaments.tournaments as _negmas_tournaments
from negmas.helpers import humanize_time
from scml.utils import anac2024_oneshot, DefaultAgentsOneShot2024
from scml_agents import get_agents

from myagent.common import LOG_ROOT, MODEL_PATH
from myagent.myagent import MyAgent

# reduces tournament logs
_real_save_stats = _negmas_tournaments.save_stats


def _save_stats_without_negotiations(*args, **kwargs):
    world = kwargs.get("world", args[0] if args else None)
    try:
        world.save_negotiations = False # type: ignore
    except Exception:
        pass
    return _real_save_stats(*args, **kwargs)


_negmas_tournaments.save_stats = _save_stats_without_negotiations


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


def derive_shard_seed(shard_seed: str, task_id: str) -> int:
    """Derive a stable per-shard RNG seed from (SHARD_SEED, array task id).

    Two benchmark runs given the SAME SHARD_SEED (e.g. two model sets submitted
    by benchmark_candidates.sh) get identical seeds per shard index, so their
    config draws — world topology, n_steps, population split, and the
    n_competitors_per_world choice — are identical and per-shard score deltas
    become meaningful (paired comparison). Different SHARD_SEEDs (or different
    shard indices) diverge.

    Uses sha256, NOT Python's hash(): the latter is salted per process
    (PYTHONHASHSEED), so it would silently break the pairing across jobs.
    """
    digest = hashlib.sha256(f"{shard_seed}:{task_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def reseed_after_import_pollution(shard_seed: str | None) -> None:
    """Reseed random/np.random after scml_agents' import-time random.seed(0).

    scml_agents has a module-level `random.seed(0)` (scml2022 oneshot team_131,
    imported transitively by get_agents/load_qualifiers) that silently fixes
    Python's global random/np.random state on import. Left alone, every fresh
    process (e.g. every sharded SLURM task) draws the identical "first" sample
    from anac2024_oneshot's config generator instead of a genuinely different
    world. This MUST be called after the polluting import and before the
    tournament call consumes the RNG.

    shard_seed=None restores real per-run randomness (OS entropy — the
    default). A non-empty shard_seed instead pins the draw deterministically
    per shard for paired set-vs-set benchmarks; see derive_shard_seed. Note
    pairing covers the CONFIG draw only, not exact world replay: MyAgent
    reseeds from OS entropy on construction (the RNG-pollution workaround in
    myagent.py) and in-world negotiation randomness differs — read paired runs
    as variance-reduced per-shard deltas, not identical scores.
    """
    if shard_seed:
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        seed = derive_shard_seed(shard_seed, task_id)
        print(f"Paired config draw: SHARD_SEED={shard_seed} task={task_id} -> seed {seed}")
        random.seed(seed)
        np.random.seed(seed)
    else:
        random.seed()
        np.random.seed()


def assignment_count(n_competitors: int, n_per_world: int) -> int:
    """Number of agent assignments a round-robin config produces.

    anac2024_oneshot with round_robin (the default) runs every C(N, k)
    competitor combination, and the assigner emits k cyclic-rotation worlds per
    combination (the max_worlds_per_config=None branch), so a single config
    yields C(N, k) * k worlds. This is the "Will run <n> different agent
    assignments" number in the tournament log — it drives per-shard wall time.
    """
    return comb(n_competitors, n_per_world) * n_per_world


def choose_competitors_per_world(
    n_competitors: int, max_assignments: int, lo: int = 2, hi: int = 4
) -> int:
    """Pick n_competitors_per_world so a shard stays within max_assignments.

    Left to itself, anac2024_oneshot draws n_competitors_per_world uniformly
    from [2, min(4, N)]. For N=11 that is 110 / 495 / 1320 assignments — the
    k=4 draw (1320) blows past any reasonable wall-time budget and times shards
    out, and note max_worlds_per_config CANNOT fix this: it caps worlds per
    competitor-combination, but the C(N, k) combination count is fixed by the
    round-robin, so the floor is exactly assignment_count(N, k).

    So we constrain k instead: among the feasible values (those whose
    assignment_count <= max_assignments) we sample uniformly, preserving some of
    the original k-diversity across shards while guaranteeing the cap. If none
    is feasible (a very large pool) we fall back to the smallest k and warn —
    the count may then exceed the cap, but nothing short of shrinking the pool
    or dropping round_robin can help there.
    """
    hi = min(hi, n_competitors)
    feasible = [
        k for k in range(lo, hi + 1)
        if assignment_count(n_competitors, k) <= max_assignments
    ]
    if not feasible:
        k = lo
        print(
            f"WARNING: no n_competitors_per_world in [{lo}, {hi}] keeps "
            f"assignments <= {max_assignments} for {n_competitors} competitors "
            f"(k={k} -> {assignment_count(n_competitors, k)}); using {k} anyway."
        )
        return k
    return random.choice(feasible)


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
    serial: bool = True,
    save_scores: str | None = None,
    max_assignments: int = 500,
    n_competitors_per_world: int | None = None,
    shard_seed: str | None = None,
    tournament_dir: str | None = None,
) -> None:
    # Which model set MyAgent will load (MODEL_DIR env var / --model-dir).
    # Fail fast here — inside the tournament a missing model surfaces as
    # hundreds of confusing per-world construction failures instead.
    model_glob = sorted(glob.glob(f"{MODEL_PATH}*Context.zip"))
    print(f"Model set: {MODEL_PATH.parent}  ({len(model_glob)} models)")
    if len(model_glob) < 6:
        raise SystemExit(
            f"Expected 6 models at {MODEL_PATH}*Context.zip, found "
            f"{len(model_glob)} — check MODEL_DIR/--model-dir (CWD: {os.getcwd()})"
        )

    if year is None:
        year = newest_agent_year()
    qualifiers, used_qualified = load_qualifiers(year)
    pool_kind = "qualifier" if used_qualified else "full-pool"
    print(f"Loaded {len(qualifiers)} {pool_kind} agents for {year}.")

    # Undo scml_agents' import-time RNG pollution; optionally pin the config
    # draw per shard for paired set-vs-set runs (see the helper's docstring).
    reseed_after_import_pollution(shard_seed)

    competitors = [MyAgent] + qualifiers
    if include_defaults:
        competitors += list(DefaultAgentsOneShot2024)

    # Fix n_competitors_per_world so a shard's assignment count stays bounded.
    # Otherwise anac2024_oneshot draws it randomly and a k=4 draw explodes to
    # C(N,4)*4 assignments (1320 for N=11), which times shards out. See
    # choose_competitors_per_world for why this — not max_worlds_per_config —
    # is the right lever.
    if n_competitors_per_world is None:
        n_competitors_per_world = choose_competitors_per_world(
            len(competitors), max_assignments
        )
    n_assignments = assignment_count(len(competitors), n_competitors_per_world)

    print(
        f"Running tournament: {len(competitors)} competitors, "
        f"n_configs={n_configs}, n_steps={n_steps}, "
        f"n_competitors_per_world={n_competitors_per_world} "
        f"(~{n_assignments} assignments/config, cap {max_assignments}) "
        f"({'serial' if serial else 'parallel'})"
    )

    # By default negmas writes the tournament's full working dir;
    #  The dir is deleted after the scores are extracted
    # — the per-world scores CSV (--save-scores) is the only artifact we keep.
    tournament_path = None
    if tournament_dir:
        run = os.environ.get("RUN_NAME", "bench")
        task = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        tournament_path = os.path.abspath(
            os.path.join(tournament_dir, f"{run}_shard{task}")
        )
        print(f"Tournament working dir: {tournament_path} (deleted afterwards)")

    start = time.perf_counter()
    results = anac2024_oneshot(
        competitors=competitors,
        verbose=True,
        n_steps=n_steps,
        n_configs=n_configs,
        n_competitors_per_world=n_competitors_per_world,
        tournament_path=tournament_path,
        # Keep world logging minimal: compact=True -> no_logs on every world;
        # forced_logs_fraction=0.0 stops negmas force-enabling FULL logs
        # (compact=False + save_negotiations=True) on every config.
        # See the save_stats wrapper at the top of this file for the third piece.
        # Scores are unaffected
        compact=True,
        forced_logs_fraction=0.0,
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

    if tournament_path and os.path.isdir(tournament_path):
        shutil.rmtree(tournament_path, ignore_errors=True)
        print(f"Removed tournament working dir {tournament_path}")

    print(f"\nFinished in {humanize_time(elapsed)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=None,
                   help="agent-pool year (default: newest available)")
    p.add_argument("--n-configs", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--include-defaults", action="store_true",
                   help="also include the weak DefaultAgentsOneShot2024 pool")
    p.add_argument("--serial", action=argparse.BooleanOptionalAction, default=True,
                   help="run the tournament serially (default). The in-process "
                        "parallel tournament deadlocks (~world 160), so pass "
                        "--no-serial only if you know you need it.")
    p.add_argument("--save-scores", default=None,
                   help="write per-agent-per-world scores to this CSV (for "
                        "sharded runs; aggregate with scripts/aggregate_benchmark.py)")
    p.add_argument("--max-assignments", type=int, default=500,
                   help="cap on agent assignments per config; n_competitors_per_world "
                        "is chosen to stay within it (default 500)")
    p.add_argument("--n-competitors-per-world", type=int, default=None,
                   help="force n_competitors_per_world (overrides --max-assignments; "
                        "default: auto-pick within the cap)")
    p.add_argument("--model-dir", default=None,
                   help="directory with the 6 mymodel<Context>.zip files MyAgent "
                        "loads (e.g. candidate_models/flex); default: "
                        "myagent/models. Equivalent to the MODEL_DIR env var; "
                        "applied by the import-time pre-scan above")
    p.add_argument("--shard-seed", default=os.environ.get("SHARD_SEED") or None,
                   help="pair the config draw across runs: same value + same "
                        "SLURM_ARRAY_TASK_ID = identical world configs, so two "
                        "model sets can be compared shard-by-shard. Default: "
                        "the SHARD_SEED env var; unset = unpaired (OS entropy)")
    p.add_argument("--tournament-dir",
                   default=os.environ.get("TOURNAMENT_DIR") or None,
                   help="base dir for negmas' tournament working files "
                        "(deleted after the scores are extracted). Default: "
                        "the TOURNAMENT_DIR env var; unset = negmas default "
                        "~/negmas/tournaments, which is NOT cleaned")
    args = p.parse_args()
    benchmark(
        year=args.year,
        n_configs=args.n_configs,
        n_steps=args.n_steps,
        include_defaults=args.include_defaults,
        serial=args.serial,
        save_scores=args.save_scores,
        max_assignments=args.max_assignments,
        n_competitors_per_world=args.n_competitors_per_world,
        shard_seed=args.shard_seed,
        tournament_dir=args.tournament_dir,
    )


if __name__ == "__main__":
    main()
