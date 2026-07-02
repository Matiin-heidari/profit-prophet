#!/bin/bash
#SBATCH --job-name=pbrs_ab
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=kisski
#SBATCH --array=0-5%3

# Multi-seed A/B: pure-profit vs pure-profit+PBRS, 3 paired seeds each.
#   tasks 0,1,2 -> pure-profit,  seeds 0,1,2
#   tasks 3,4,5 -> PBRS,         seeds 0,1,2
# Same seed across arms => same network init => a fair paired comparison.
# Compare 0_key/* curves as mean +/- spread over the 3 seeds per arm.
# (%3 = run 3 tasks at a time; raise/lower to fit the cluster.)

set -euo pipefail
module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"
cd "$SLURM_SUBMIT_DIR"

ARMS=(pureprofit pureprofit pureprofit pbrs pbrs pbrs)
SEEDS=(0 1 2 0 1 2)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
export SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Shared config (identical for both arms).
export TRAIN_CONTEXTS="StrongSupplierContext,WeakSupplierContext"
export EVAL_FREQ=40000
export N_EVAL_EPISODES=10
export LOG_REWARD_COMPONENTS=1

# Reward: pure profit only (score_delta), all hand-shaping off.
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

# The ONLY difference between arms: PBRS potential term on/off.
if [ "$ARM" = "pbrs" ]; then
  export REWARD_POTENTIAL_WEIGHT=0.5
else
  export REWARD_POTENTIAL_WEIGHT=0.0
fi

export RUN_NAME="ab_${ARM}_s${SEED}"

echo "=== ${RUN_NAME}  (array task ${SLURM_ARRAY_TASK_ID}) ==="
echo "ARM=${ARM}  SEED=${SEED}  REWARD_POTENTIAL_WEIGHT=${REWARD_POTENTIAL_WEIGHT}"
hostname; date
git rev-parse --short HEAD; git branch --show-current

python -m myagent.train 400000

echo "=== Done ${RUN_NAME} ==="
date
