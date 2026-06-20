#!/bin/bash
#SBATCH --job-name=reward_validate
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-17%1
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"

cd "$SLURM_SUBMIT_DIR"

export TRAIN_CONTEXTS=BalancedSupplierContext
export N_EVAL_EPISODES=5
export EVAL_FREQ=50000
export DIAGNOSTICS_FREQ=10000
export LOG_REWARD_COMPONENTS=0

export REWARD_SCORE_DELTA_WEIGHT=0.1
export REWARD_NEED_WEIGHT=0.0
export REWARD_SHORTFALL_WEIGHT=0.0
export REWARD_OVERSHOOT_WEIGHT=0.0
export REWARD_DISPOSAL_WEIGHT=0.0
export REWARD_PRODUCTIVITY_WEIGHT=0.0
export REWARD_TIME_PRESSURE_WEIGHT=1.0
export REWARD_NEED_NORMALIZER=0.0

CONFIG_ID=$((SLURM_ARRAY_TASK_ID / 3))
REPEAT_ID=$((SLURM_ARRAY_TASK_ID % 3))

case "$CONFIG_ID" in
  0)
    export RUN_NAME=validate_bs_00_baseline_r${REPEAT_ID}
    ;;
  1)
    export RUN_NAME=validate_bs_01_need_0001_shortfall_0001_r${REPEAT_ID}
    export REWARD_NEED_WEIGHT=0.0001
    export REWARD_SHORTFALL_WEIGHT=0.0001
    ;;
  2)
    export RUN_NAME=validate_bs_02_need_00025_shortfall_0001_r${REPEAT_ID}
    export REWARD_NEED_WEIGHT=0.00025
    export REWARD_SHORTFALL_WEIGHT=0.0001
    ;;
  3)
    export RUN_NAME=validate_bs_03_score_delta_05_r${REPEAT_ID}
    export REWARD_SCORE_DELTA_WEIGHT=0.5
    ;;
  4)
    export RUN_NAME=validate_bs_04_productivity_001_r${REPEAT_ID}
    export REWARD_PRODUCTIVITY_WEIGHT=0.001
    ;;
  5)
    export RUN_NAME=validate_bs_05_overshoot_0001_disposal_0001_r${REPEAT_ID}
    export REWARD_OVERSHOOT_WEIGHT=0.0001
    export REWARD_DISPOSAL_WEIGHT=0.0001
    ;;
  *)
    echo "Unknown CONFIG_ID=$CONFIG_ID"
    exit 1
    ;;
esac

echo "=== Validation sweep config ==="
echo "task_id=$SLURM_ARRAY_TASK_ID"
echo "config_id=$CONFIG_ID"
echo "repeat_id=$REPEAT_ID"
echo "RUN_NAME=$RUN_NAME"
echo "TRAIN_CONTEXTS=$TRAIN_CONTEXTS"
echo "N_EVAL_EPISODES=$N_EVAL_EPISODES"
echo "REWARD_SCORE_DELTA_WEIGHT=$REWARD_SCORE_DELTA_WEIGHT"
echo "REWARD_NEED_WEIGHT=$REWARD_NEED_WEIGHT"
echo "REWARD_SHORTFALL_WEIGHT=$REWARD_SHORTFALL_WEIGHT"
echo "REWARD_OVERSHOOT_WEIGHT=$REWARD_OVERSHOOT_WEIGHT"
echo "REWARD_DISPOSAL_WEIGHT=$REWARD_DISPOSAL_WEIGHT"
echo "REWARD_PRODUCTIVITY_WEIGHT=$REWARD_PRODUCTIVITY_WEIGHT"

python -m myagent.train
