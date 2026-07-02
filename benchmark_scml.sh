#!/bin/bash
#SBATCH --job-name=scml_benchmark
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=kisski

# Runs scripts/benchmark.py: MyAgent (loads its 6 per-context models) in an
# ANAC-style OneShot tournament against last year's qualifiers, reporting
# MyAgent's rank / gap-to-best and the per-agent spread (noise floor).
# No training here — it benchmarks whatever models are currently on disk.
# Override knobs at submit time, e.g.:
#   N_CONFIGS=20 N_STEPS=50 sbatch benchmark_scml.sh

set -euo pipefail
module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"
cd "$SLURM_SUBMIT_DIR"

YEAR="${YEAR:-2024}"
N_CONFIGS="${N_CONFIGS:-10}"
N_STEPS="${N_STEPS:-50}"

# Group this benchmark's per-world context-usage logs under one run dir so
# benchmark.py can report the routing / fallback breakdown at the end.
export RUN_NAME="${RUN_NAME:-benchmark_${SLURM_JOB_ID:-local}}"
export LOG_CONTEXT_USAGE="${LOG_CONTEXT_USAGE:-1}"

echo "=== Host ==="; hostname
echo "=== Date ==="; date
echo "=== Python ==="; which python; python --version
echo "=== Git ==="; git rev-parse --short HEAD; git branch --show-current
echo "=== Benchmark config ==="
echo "YEAR=${YEAR}  N_CONFIGS=${N_CONFIGS}  N_STEPS=${N_STEPS}  INCLUDE_DEFAULTS=${INCLUDE_DEFAULTS:-0}  RUN_NAME=${RUN_NAME}"

EXTRA=""
if [ "${INCLUDE_DEFAULTS:-0}" != "0" ]; then
  EXTRA="--include-defaults"
fi

echo "=== Run benchmark ==="
python scripts/benchmark.py \
  --year "${YEAR}" \
  --n-configs "${N_CONFIGS}" \
  --n-steps "${N_STEPS}" \
  ${EXTRA}

echo "=== Done ==="; date
