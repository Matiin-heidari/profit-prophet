#!/bin/bash
#SBATCH --job-name=scml_pureprofit
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=kisski

module load miniforge3/24.3.0-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $PROJECT_DIR/profit-prophet-env/agentic

cd "$SLURM_SUBMIT_DIR"

# --- PURE-PROFIT CONTROL ---------------------------------------------------
# Tests whether optimizing the TRUE objective (score = balance/initial_balance)
# improves eval performance, with ALL reward shaping disabled. Setting these env
# vars overrides the per-context default weight table for every context.
#
# If eval score/rank improves -> shaping was the problem (tune from here).
# If eval score is flat/declines too -> the bottleneck is learnability/benchmark,
# not the shaping. Either way this is the decisive diagnostic.
#
# score_delta is the only active term, so its weight is just a global reward
# scale (PPO normalizes advantages); 10.0 keeps it out of tiny-number territory.
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

# Match adj_reward_400k so the runs are directly comparable.
export TRAIN_CONTEXTS="${TRAIN_CONTEXTS:-StrongSupplierContext,WeakSupplierContext}"
export RUN_NAME="${RUN_NAME:-pure_profit_400k}"
export EVAL_FREQ="${EVAL_FREQ:-40000}"
export N_EVAL_EPISODES="${N_EVAL_EPISODES:-10}"
export LOG_REWARD_COMPONENTS=1

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

echo "=== Run config (pure-profit control) ==="
echo "RUN_NAME=${RUN_NAME}"
echo "TRAIN_CONTEXTS=${TRAIN_CONTEXTS}"
echo "EVAL_FREQ=${EVAL_FREQ}"
echo "N_EVAL_EPISODES=${N_EVAL_EPISODES}"
echo "REWARD_SCORE_DELTA_WEIGHT=${REWARD_SCORE_DELTA_WEIGHT}  (all shaping weights = 0)"

echo "=== Torch ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "=== Start training (400k, pure-profit reward) ==="
python -m myagent.train 400000
echo "=== Done ==="
date
