#!/bin/bash
#SBATCH --job-name=scml_seeds
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=kisski
#SBATCH --array=0-2

# Like train_scml.sh (current setup: pure-profit reward, all 6 contexts), but
# trains 3 SEEDS — one array task per seed (seed = array task id). One submit:
#   sbatch train_scml_seeds.sh
#
# Re-run a seed with e.g.
#   sbatch --array=1 train_scml_seeds.sh
# Seeded models -> myagent/models/mymodel<ctx>_seeds_s<seed>_seed<seed>.zip
# (canonical mymodel<ctx>.zip untouched). Curves -> log/tensorboard_logs/seeds_s<seed>/.
# Optional: STEPS=200000 sbatch train_scml_seeds.sh
set -euo pipefail
module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"
cd "$SLURM_SUBMIT_DIR"

export SEED=$SLURM_ARRAY_TASK_ID
export RUN_NAME="${RUN_NAME:-seeds}_s${SEED}"
export EVAL_FREQ="${EVAL_FREQ:-40000}"
export N_EVAL_EPISODES="${N_EVAL_EPISODES:-10}"
STEPS="${STEPS:-400000}"
# TRAIN_CONTEXTS unset => all 6 contexts (override to train a subset).

# --- REWARD: PURE-PROFIT (current setup; matches train_scml.sh) -------------
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

echo "=== SEED=${SEED}  (array task ${SLURM_ARRAY_TASK_ID}) ==="
echo "=== Host ==="; hostname
echo "=== Date ==="; date
echo "=== Python ==="; which python; python --version
echo "=== Git ==="; git rev-parse --short HEAD; git branch --show-current

echo "=== Run config ==="
echo "SEED=${SEED}  RUN_NAME=${RUN_NAME}  STEPS=${STEPS:-<default>}"
echo "TRAIN_CONTEXTS=${TRAIN_CONTEXTS:-all}  EVAL_FREQ=${EVAL_FREQ}  N_EVAL_EPISODES=${N_EVAL_EPISODES}"
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

echo "=== Start training (seed ${SEED}, all contexts) ==="
python -m myagent.train ${STEPS}
echo "=== Done seed ${SEED} ==="
date
