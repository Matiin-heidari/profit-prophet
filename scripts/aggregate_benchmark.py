"""Aggregate sharded benchmark scores into one ranking.

Each shard (see slurm/benchmark_shard.sh) runs a small serial tournament and
writes its per-agent-per-world scores via `benchmark.py --save-scores`. Because
scoring is just an average over worlds, concatenating the shards and re-ranking is
statistically equivalent to one big run — but it sidesteps the in-process parallel
tournament deadlock by using the SLURM scheduler for parallelism instead.

Usage:
    python scripts/aggregate_benchmark.py <dir-with-shard_*.csv>
    python scripts/aggregate_benchmark.py <dir> --run <RUN_NAME>   # also show context usage
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.benchmark import report_ranking, report_context_usage


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scores_dir", help="directory containing shard_*.csv")
    p.add_argument("--pattern", default="shard_*.csv",
                   help="glob for shard score files (default: shard_*.csv)")
    p.add_argument("--run", default=None,
                   help="RUN_NAME to also aggregate context-usage logs for")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.scores_dir, args.pattern)))
    if not files:
        print(f"No shard files matching {args.pattern} in {args.scores_dir}")
        sys.exit(1)

    frames = [pd.read_csv(f) for f in files]
    scores = pd.concat(frames, ignore_index=True)
    print(f"Aggregated {len(files)} shards -> {len(scores)} agent-world scores "
          f"({scores['agent_type'].nunique()} agents).")

    report_ranking(scores)

    if args.run:
        report_context_usage(args.run)


if __name__ == "__main__":
    main()
