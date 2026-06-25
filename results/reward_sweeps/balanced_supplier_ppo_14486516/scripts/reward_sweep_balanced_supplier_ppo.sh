#!/bin/bash
#SBATCH --job-name=ppo_sweep
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-15%1
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"

cd "$SLURM_SUBMIT_DIR"

export TRAIN_CONTEXTS=BalancedSupplierContext
export N_EVAL_EPISODES=3
export EVAL_FREQ=50000
export DIAGNOSTICS_FREQ=10000
export LOG_REWARD_COMPONENTS=1

# Defaults: baseline-compatible unless overwritten below.
export REWARD_SCORE_DELTA_WEIGHT=0.1
export REWARD_NEED_WEIGHT=0.0
export REWARD_SHORTFALL_WEIGHT=0.0
export REWARD_OVERSHOOT_WEIGHT=0.0
export REWARD_DISPOSAL_WEIGHT=0.0
export REWARD_PRODUCTIVITY_WEIGHT=0.0
export REWARD_TIME_PRESSURE_WEIGHT=1.0
export REWARD_NEED_NORMALIZER=0.0

case "$SLURM_ARRAY_TASK_ID" in
  0)
    export RUN_NAME=ppo_sweep_bs_00_baseline
    ;;
  1)
    export RUN_NAME=ppo_sweep_bs_01_default_only
    export REWARD_SCORE_DELTA_WEIGHT=0.0
    ;;
  2)
    export RUN_NAME=ppo_sweep_bs_02_score_delta_05
    export REWARD_SCORE_DELTA_WEIGHT=0.5
    ;;
  3)
    export RUN_NAME=ppo_sweep_bs_03_need_0001
    export REWARD_NEED_WEIGHT=0.0001
    ;;
  4)
    export RUN_NAME=ppo_sweep_bs_04_need_00025
    export REWARD_NEED_WEIGHT=0.00025
    ;;
  5)
    export RUN_NAME=ppo_sweep_bs_05_need_0005
    export REWARD_NEED_WEIGHT=0.0005
    ;;
  6)
    export RUN_NAME=ppo_sweep_bs_06_shortfall_0001
    export REWARD_SHORTFALL_WEIGHT=0.0001
    ;;
  7)
    export RUN_NAME=ppo_sweep_bs_07_need_0001_shortfall_0001
    export REWARD_NEED_WEIGHT=0.0001
    export REWARD_SHORTFALL_WEIGHT=0.0001
    ;;
  8)
    export RUN_NAME=ppo_sweep_bs_08_need_00025_shortfall_0001
    export REWARD_NEED_WEIGHT=0.00025
    export REWARD_SHORTFALL_WEIGHT=0.0001
    ;;
  9)
    export RUN_NAME=ppo_sweep_bs_09_productivity_0005
    export REWARD_PRODUCTIVITY_WEIGHT=0.0005
    ;;
  10)
    export RUN_NAME=ppo_sweep_bs_10_productivity_001
    export REWARD_PRODUCTIVITY_WEIGHT=0.001
    ;;
  11)
    export RUN_NAME=ppo_sweep_bs_11_need_0001_productivity_0005
    export REWARD_NEED_WEIGHT=0.0001
    export REWARD_PRODUCTIVITY_WEIGHT=0.0005
    ;;
  12)
    export RUN_NAME=ppo_sweep_bs_12_need_0001_shortfall_0001_productivity_0005
    export REWARD_NEED_WEIGHT=0.0001
    export REWARD_SHORTFALL_WEIGHT=0.0001
    export REWARD_PRODUCTIVITY_WEIGHT=0.0005
    ;;
  13)
    export RUN_NAME=ppo_sweep_bs_13_need_00025_shortfall_0001_productivity_0005
    export REWARD_NEED_WEIGHT=0.00025
    export REWARD_SHORTFALL_WEIGHT=0.0001
    export REWARD_PRODUCTIVITY_WEIGHT=0.0005
    ;;
  14)
    export RUN_NAME=ppo_sweep_bs_14_overshoot_0001_disposal_0001
    export REWARD_OVERSHOOT_WEIGHT=0.0001
    export REWARD_DISPOSAL_WEIGHT=0.0001
    ;;
  15)
    export RUN_NAME=ppo_sweep_bs_15_need_0001_overshoot_0001_disposal_0001
    export REWARD_NEED_WEIGHT=0.0001
    export REWARD_OVERSHOOT_WEIGHT=0.0001
    export REWARD_DISPOSAL_WEIGHT=0.0001
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

echo "=== Sweep config ==="
echo "task_id=$SLURM_ARRAY_TASK_ID"
echo "RUN_NAME=$RUN_NAME"
echo "TRAIN_CONTEXTS=$TRAIN_CONTEXTS"
echo "REWARD_SCORE_DELTA_WEIGHT=$REWARD_SCORE_DELTA_WEIGHT"
echo "REWARD_NEED_WEIGHT=$REWARD_NEED_WEIGHT"
echo "REWARD_SHORTFALL_WEIGHT=$REWARD_SHORTFALL_WEIGHT"
echo "REWARD_OVERSHOOT_WEIGHT=$REWARD_OVERSHOOT_WEIGHT"
echo "REWARD_DISPOSAL_WEIGHT=$REWARD_DISPOSAL_WEIGHT"
echo "REWARD_PRODUCTIVITY_WEIGHT=$REWARD_PRODUCTIVITY_WEIGHT"
echo "REWARD_TIME_PRESSURE_WEIGHT=$REWARD_TIME_PRESSURE_WEIGHT"
echo "REWARD_NEED_NORMALIZER=$REWARD_NEED_NORMALIZER"

python -m myagent.train
