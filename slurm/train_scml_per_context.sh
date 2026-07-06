#!/bin/bash
#SBATCH --job-name=scml_ctx
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=kisski
#SBATCH --array=0-5

# Train all 6 contexts as SEPARATE array jobs (one context per task) instead of
# one job that trains them together. Each task trains a single context and saves
# its canonical model (myagent/models/mymodel<ctx>.zip) at the end, so a
# wall-clock timeout only loses the context(s) still running — every context
# that finished is already on disk. Re-run just the failed indices with, e.g.:
#   sbatch --array=2,5 slurm/train_scml_per_context.sh
#
# Knobs (override at submit time):
#   STEPS=200000 sbatch slurm/train_scml_per_context.sh
#   sbatch --array=0-5%3 slurm/train_scml_per_context.sh   # cap to 3 at a time
set -euo pipefail
module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"
cd "$SLURM_SUBMIT_DIR"

# Order matches ALL_CONTEXTS in myagent/common.py (index == array task id).
CONTEXTS=(
  StrongSupplierContext
  BalancedSupplierContext
  WeakSupplierContext
  StrongConsumerContext
  BalancedConsumerContext
  WeakConsumerContext
)
CONTEXT=${CONTEXTS[$SLURM_ARRAY_TASK_ID]}

# This task trains ONLY its one context -> all CPUs go to that single model.
export TRAIN_CONTEXTS="$CONTEXT"
export RUN_NAME="${RUN_NAME:-per_context}_${CONTEXT}"
export EVAL_FREQ="${EVAL_FREQ:-40000}"
export N_EVAL_EPISODES="${N_EVAL_EPISODES:-10}"
STEPS="${STEPS:-400000}"

# --- REWARD: PURE-PROFIT (current best; matches train_scml.sh) --------------
# score_delta only; all hand-shaping off (see CLAUDE.md §4). Overrides the
# per-context default weight table.
export REWARD_SCORE_DELTA_WEIGHT=100.0
export REWARD_NEED_WEIGHT=0.0
export REWARD_SHORTFALL_WEIGHT=0.0
export REWARD_OVERSHOOT_WEIGHT=0.0
export REWARD_DISPOSAL_WEIGHT=0.0
export REWARD_PRODUCTIVITY_WEIGHT=0.0
export REWARD_TIME_PRESSURE_WEIGHT=0.0
export REWARD_NEED_NORMALIZER=0.0
export REWARD_PRICE_WEIGHT=0.0
export REWARD_DEAL_WEIGHT=0.0
export REWARD_ENGAGEMENT_WEIGHT=0.0
export REWARD_MARGIN_WEIGHT=0.0
export REWARD_POTENTIAL_WEIGHT=0.0

echo "=== ${CONTEXT}  (array task ${SLURM_ARRAY_TASK_ID}) ==="
echo "=== Host ==="; hostname
echo "=== Date ==="; date
echo "=== Python ==="; which python; python --version
echo "=== Git ==="; git rev-parse --short HEAD; git branch --show-current
echo "=== Run config ==="
echo "CONTEXT=${CONTEXT}  STEPS=${STEPS}  RUN_NAME=${RUN_NAME}"
echo "REWARD=pure-profit  REWARD_SCORE_DELTA_WEIGHT=${REWARD_SCORE_DELTA_WEIGHT}  (all shaping = 0)"

echo "=== Torch ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "=== Start training (${STEPS} steps, ${CONTEXT}) ==="
python -m myagent.train "${STEPS}"
echo "=== Done ${CONTEXT} ==="
date
