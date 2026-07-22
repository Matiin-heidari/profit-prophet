#!/bin/bash
#SBATCH --job-name=ppo_bs_scratch
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-2%1
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

export TRAIN_CONTEXTS=BalancedSupplierContext
export SEED="$SLURM_ARRAY_TASK_ID"
export RUN_NAME="ppo_fromscratch_bs_s${SEED}"

export WARM_START=0

export EVAL_FREQ=40000
export N_EVAL_EPISODES=10
export DIAGNOSTICS_FREQ=10000
export LOG_REWARD_COMPONENTS=0

echo "=== PPO from scratch BalancedSupplier ==="
echo "RUN_NAME=$RUN_NAME"
echo "SEED=$SEED"
echo "TRAIN_CONTEXTS=$TRAIN_CONTEXTS"
echo "WARM_START=$WARM_START"

python -m myagent.train 400000
