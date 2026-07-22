#!/bin/bash
#SBATCH --job-name=bc_equal
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=kisski
#SBATCH --array=0-5%2

set -euo pipefail

module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

CONTEXTS=(
  StrongSupplierContext
  BalancedSupplierContext
  WeakSupplierContext
  StrongConsumerContext
  BalancedConsumerContext
  WeakConsumerContext
)

CONTEXT="${CONTEXTS[$SLURM_ARRAY_TASK_ID]}"

export LOG_REWARD_COMPONENTS=0
export RUN_NAME="${RUN_NAME:-bc_equal_s${SEED:-0}}"

N_SAMPLES="${N_SAMPLES:-50000}"
BC_EPOCHS="${BC_EPOCHS:-8}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-512}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-myagent/models/bc_equal}"

echo "=== BC EqualDist-style pretraining ==="
echo "task_id=$SLURM_ARRAY_TASK_ID"
echo "context=$CONTEXT"
echo "run_name=$RUN_NAME"
echo "n_samples=$N_SAMPLES"
echo "bc_epochs=$BC_EPOCHS"
echo "batch_size=$BC_BATCH_SIZE"
echo "seed=$SEED"
echo "output_dir=$OUTPUT_DIR"
echo "PYTHONPATH=$PYTHONPATH"

echo "=== Host ==="
hostname

echo "=== Date ==="
date

echo "=== Python ==="
which python
python --version

echo "=== Git ==="
git rev-parse --short HEAD
git branch --show-current

python -m py_compile scripts/bc_pretrain_expert.py myagent/train.py myagent/common.py

python scripts/bc_pretrain_expert.py \
  --expert equal \
  --contexts "$CONTEXT" \
  --n-samples "$N_SAMPLES" \
  --bc-epochs "$BC_EPOCHS" \
  --batch-size "$BC_BATCH_SIZE" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME"

echo "=== Done ==="
date
