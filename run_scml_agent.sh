#!/bin/bash
#SBATCH --job-name=scml_agent
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=kisski

module load miniforge3/24.3.0-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $PROJECT_DIR/envs/agentic

cd $PROJECT_DIR/repos/profit-prophet

# The runtime agent should always load all models.
unset TRAIN_CONTEXTS

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
echo "TRAIN_CONTEXTS=${TRAIN_CONTEXTS:-unset}"

echo "=== Start agent evaluation ==="
python -m myagent.myagent

echo "=== Done ==="
date
