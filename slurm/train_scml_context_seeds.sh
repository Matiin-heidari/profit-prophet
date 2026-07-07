#!/bin/bash
#SBATCH --job-name=scml_ctxseed
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=kisski
#SBATCH --array=0-17%3

# Multi-seed test of the PER-CONTEXT reward functions.
# 6 contexts x 3 seeds = 18 tasks
#
# Task layout:  context = task_id % 6,  seed = task_id / 6
#   tasks  0-5  -> seed 0 (all 6 contexts)
#   tasks  6-11 -> seed 1
#   tasks 12-17 -> seed 2
#
# ARM selects the reward under test (paired seeds => fair A/B). Neither arm uses
# margin:
#   ARM=premargin   (default) the shaping reward BEFORE margin (git 16f1347~1):
#                             per-context price/deal + need floor, NO margin
#   ARM=pureprofit            score_delta only (the baseline to beat)
# Run each arm with the same seeds, then compare 0_key/* per (context, seed):
#   sbatch slurm/train_scml_context_seeds.sh                 # premargin shaping
#   ARM=pureprofit sbatch slurm/train_scml_context_seeds.sh  # baseline
# Concurrency is capped at 3 at a time (%3) — each task is 48 CPUs / 96G, so 3
# concurrent ≈ 144 CPUs (matches train_scml_pbrs_ab.sh). Override at submit time,
# e.g. `sbatch --array=0-17%6 slurm/train_scml_context_seeds.sh`.
set -euo pipefail
module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"
cd "$SLURM_SUBMIT_DIR"

# Order matches ALL_CONTEXTS in myagent/common.py.
CONTEXTS=(
  StrongSupplierContext
  BalancedSupplierContext
  WeakSupplierContext
  StrongConsumerContext
  BalancedConsumerContext
  WeakConsumerContext
)
CONTEXT=${CONTEXTS[$(( SLURM_ARRAY_TASK_ID % 6 ))]}
export SEED=$(( SLURM_ARRAY_TASK_ID / 6 ))

ARM="${ARM:-premargin}"
# Same seed across arms => same network init/env => a fair paired comparison.
export TRAIN_CONTEXTS="$CONTEXT"
export RUN_NAME="ab_${ARM}_s${SEED}"   # -> log/tensorboard_logs/ab_<arm>_s<seed>/<context>/
export EVAL_FREQ="${EVAL_FREQ:-40000}"
export N_EVAL_EPISODES="${N_EVAL_EPISODES:-10}"
export LOG_REWARD_COMPONENTS="${LOG_REWARD_COMPONENTS:-1}"
STEPS="${STEPS:-400000}"

if [ "$ARM" = "pureprofit" ]; then
  # Baseline: score_delta only, all hand-shaping off.
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
elif [ "$ARM" = "premargin" ]; then
  # The shaping reward that preceded margin (git 16f1347~1): per-context
  # price/deal + a small need floor, no margin. score_delta=3.0.
  export REWARD_SCORE_DELTA_WEIGHT=3.0
  export REWARD_TIME_PRESSURE_WEIGHT=0.0
  export REWARD_NEED_WEIGHT=0.05
  export REWARD_MARGIN_WEIGHT=0.0        # kill the current default-table margin
  export REWARD_SHORTFALL_WEIGHT=0.0
  export REWARD_OVERSHOOT_WEIGHT=0.0
  export REWARD_DISPOSAL_WEIGHT=0.0
  export REWARD_PRODUCTIVITY_WEIGHT=0.0
  export REWARD_ENGAGEMENT_WEIGHT=0.0
  export REWARD_POTENTIAL_WEIGHT=0.0
  case "$CONTEXT" in
    Strong*)   export REWARD_PRICE_WEIGHT=0.50; export REWARD_DEAL_WEIGHT=0.00 ;;
    Balanced*) export REWARD_PRICE_WEIGHT=0.30; export REWARD_DEAL_WEIGHT=0.00 ;;
    Weak*)     export REWARD_PRICE_WEIGHT=0.00; export REWARD_DEAL_WEIGHT=0.20 ;;
  esac
else
  # Only premargin/pureprofit are supported. Refuse anything else rather than
  # silently falling back to train.py's default table, which includes margin.
  echo "ERROR: unknown ARM='${ARM}' (expected 'premargin' or 'pureprofit')" >&2
  exit 1
fi

echo "=== ${CONTEXT}  ARM=${ARM}  SEED=${SEED}  (array task ${SLURM_ARRAY_TASK_ID}) ==="
echo "=== Host ==="; hostname
echo "=== Date ==="; date
echo "=== Python ==="; which python; python --version
echo "=== Git ==="; git rev-parse --short HEAD; git branch --show-current
echo "=== Run config ==="
echo "CONTEXT=${CONTEXT}  ARM=${ARM}  SEED=${SEED}  STEPS=${STEPS}  RUN_NAME=${RUN_NAME}"

echo "=== Torch ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "=== Start training (${STEPS} steps, ${CONTEXT}, ARM=${ARM}, seed ${SEED}) ==="
# Pin the pre-2026-07-07 action space: this script reproduces arms that were
# defined on FlexibleActionManager (the in-code default is now "accept").
export ACTION_MANAGER="${ACTION_MANAGER:-flexible}"
python -m myagent.train "${STEPS}"
echo "=== Done ${CONTEXT} ARM=${ARM} seed ${SEED} ==="
date
