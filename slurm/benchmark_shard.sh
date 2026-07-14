#!/bin/bash
#SBATCH --job-name=scml_bench_shard
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=kisski
#SBATCH --array=0-4%5

# SHARDED benchmark: parallelize across SLURM tasks instead of within one process.
# The in-process parallel tournament DEADLOCKS (negmas ProcessPoolExecutor, ~world
# 160) and a single serial run wastes a whole node on one core. Instead, each array
# task runs ONE config SERIALLY on a small (4-CPU) allocation; N tasks run at once,
# so the full N-config benchmark finishes in ~one config's time (~70 min) using
# ~N*4 cores total. Each shard saves its per-agent-per-world scores; aggregate them
# into one ranking afterwards.
#
#   sbatch slurm/benchmark_shard.sh                 # 5 shards = 5 configs
#   sbatch --array=0-9%10 slurm/benchmark_shard.sh  # 10 configs
# Then (after the array finishes) aggregate the shards into one ranking — the last
# line each task prints gives you the exact command, i.e.:
#   python scripts/aggregate_benchmark.py log/benchmark_shards/<RUN_NAME> --run <RUN_NAME>
set -euo pipefail
module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"
cd "$SLURM_SUBMIT_DIR"

YEAR="${YEAR:-2024}"
N_STEPS="${N_STEPS:-50}"
# Optional alternative model set (e.g. candidate_models/flex), relative to the
# repo root. Empty = the canonical myagent/models. Exported so MyAgent picks it
# up in every process of this task (myagent/common.py reads MODEL_DIR).
MODEL_DIR="${MODEL_DIR:-}"
export MODEL_DIR
# Optional paired config draw: submit two arrays with the SAME SHARD_SEED (e.g.
# two model sets) and shard i of both draws the IDENTICAL world config, so
# per-shard score deltas are meaningful. Empty = unpaired (OS entropy, the
# historical behavior). benchmark.py derives the per-shard seed from
# (SHARD_SEED, SLURM_ARRAY_TASK_ID) via sha256. NOTE: pairing covers the config
# draw only (topology, n_steps, k) — in-world randomness still differs, so read
# results as variance-reduced paired deltas, not identical scores.
SHARD_SEED="${SHARD_SEED:-}"
export SHARD_SEED
# negmas' tournament working dir (full per-world logs) — kept OFF the small
# home quota (~/negmas/tournaments overflowed 2026-07-13, killing shards with
# "Disk quota exceeded") and deleted by benchmark.py after the scores CSV is
# saved. Repo-relative -> lands on the project filesystem.
TOURNAMENT_DIR="${TOURNAMENT_DIR:-log/negmas_tournaments}"
export TOURNAMENT_DIR
# One shared run name across all shards of this array (SLURM_ARRAY_JOB_ID is the
# array's parent id, identical for every task) so scores + context-usage logs
# collect into one place.
RUN_NAME="${RUN_NAME:-bench_shard_${SLURM_ARRAY_JOB_ID:-local}}"
SCORES_DIR="log/benchmark_shards/${RUN_NAME}"
mkdir -p "$SCORES_DIR"
export RUN_NAME
export LOG_CONTEXT_USAGE=1

echo "=== shard ${SLURM_ARRAY_TASK_ID}  RUN_NAME=${RUN_NAME}  YEAR=${YEAR}  N_STEPS=${N_STEPS}  MODEL_DIR=${MODEL_DIR:-myagent/models (default)}  SHARD_SEED=${SHARD_SEED:-unpaired} ==="
hostname; date
git rev-parse --short HEAD; git branch --show-current

python scripts/benchmark.py \
  --year "${YEAR}" \
  --n-configs 1 \
  --n-steps "${N_STEPS}" \
  --serial \
  --save-scores "${SCORES_DIR}/shard_${SLURM_ARRAY_TASK_ID}.csv"

echo "=== shard ${SLURM_ARRAY_TASK_ID} done ==="
echo "When ALL shards finish, aggregate with:"
echo "  python scripts/aggregate_benchmark.py ${SCORES_DIR} --run ${RUN_NAME}"
date
