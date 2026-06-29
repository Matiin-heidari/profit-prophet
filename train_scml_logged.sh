#!/bin/bash
#SBATCH --job-name=scml_train_logged
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=kisski

module load miniforge3/24.3.0-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $PROJECT_DIR/envs/agentic

cd "$SLURM_SUBMIT_DIR"

export LOG_REWARD_COMPONENTS=1
export RUN_NAME="${RUN_NAME:-logged_100k}"


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
echo "LOG_REWARD_COMPONENTS=${LOG_REWARD_COMPONENTS:-0}"
echo "LOG_WORLD=${LOG_WORLD:-0}"

echo "=== Torch ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "=== Start training (100k steps, logging on) ==="
python -m myagent.train 100000
echo "=== Done ==="
date
