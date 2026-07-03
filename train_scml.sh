#!/bin/bash
#SBATCH --job-name=scml_train
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

echo "=== Run config ==="
echo "RUN_NAME=${RUN_NAME:-default}"
echo "TRAIN_CONTEXTS=${TRAIN_CONTEXTS:-all}"
echo "EVAL_FREQ=${EVAL_FREQ:-auto}"
echo "N_EVAL_EPISODES=${N_EVAL_EPISODES:-3}"
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

echo "=== Start training ==="
python -m myagent.train
echo "=== Done ==="
date
