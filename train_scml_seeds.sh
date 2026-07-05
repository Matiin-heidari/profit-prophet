#!/bin/bash
#SBATCH --job-name=scml_seeds
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=kisski
#SBATCH --array=0-17%3

# Like train_scml.sh, but 3 SEEDS x 6 contexts
# as SEPARATE tasks (one context+seed per task) so each model gets all 48 CPUs and
# a timeout only loses the task still running. One submit:
#   sbatch train_scml_seeds.sh
#
# Task layout:  context = task_id % 6,  seed = task_id / 6
#   tasks  0-5  -> seed 0 (all 6 contexts)
#   tasks  6-11 -> seed 1
#   tasks 12-17 -> seed 2
# %3 = at most 3 running at once (3 x 48 = 144 CPUs). Re-run failures with e.g.
#   sbatch --array=4,17 slurm/train_scml_seeds.sh
# Seeded models -> myagent/models/mymodel<ctx>_seeds_s<seed>_seed<seed>.zip
# Curves -> log/tensorboard_logs/seeds_s<seed>/<ctx>/.
set -euo pipefail
module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"
cd "$SLURM_SUBMIT_DIR"

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

# This task trains ONLY its one context
export TRAIN_CONTEXTS="$CONTEXT"
export RUN_NAME="${RUN_NAME:-seeds}_s${SEED}"
export EVAL_FREQ="${EVAL_FREQ:-40000}"
export N_EVAL_EPISODES="${N_EVAL_EPISODES:-10}"
STEPS="${STEPS:-400000}"

# --- REWARD ---
export REWARD_SCORE_DELTA_WEIGHT=10.0
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

echo "=== ${CONTEXT}  SEED=${SEED}  (array task ${SLURM_ARRAY_TASK_ID}) ==="
echo "=== Host ==="; hostname
echo "=== Date ==="; date
echo "=== Python ==="; which python; python --version
echo "=== Git ==="; git rev-parse --short HEAD; git branch --show-current

echo "=== Run config ==="
echo "CONTEXT=${CONTEXT}  SEED=${SEED}  RUN_NAME=${RUN_NAME}  STEPS=${STEPS:-<default>}"
echo "EVAL_FREQ=${EVAL_FREQ}  N_EVAL_EPISODES=${N_EVAL_EPISODES}"
echo "REWARD=pure-profit  REWARD_SCORE_DELTA_WEIGHT=${REWARD_SCORE_DELTA_WEIGHT}  (all shaping weights = 0)"

echo "=== Torch ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "=== Start training (${CONTEXT}, seed ${SEED}) ==="
python -m myagent.train ${STEPS}
echo "=== Done ${CONTEXT} seed ${SEED} ==="
date
