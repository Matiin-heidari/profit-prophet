#!/bin/bash
#SBATCH --job-name=scml_test
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=kisski-h100
#SBATCH --gres=gpu:1

module load miniforge3/24.3.0-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $PROJECT_DIR/envs/agentic

cd $PROJECT_DIR/repos/profit-prophet

echo "=== Host ==="
hostname
echo "=== Date ==="
date
echo "=== Python ==="
which python
python --version
echo "=== Torch/CUDA ==="
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
