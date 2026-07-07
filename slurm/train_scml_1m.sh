#!/bin/bash
#SBATCH --job-name=scml_1m
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --partition=kisski
#SBATCH --array=0-2

# LONG RUN (default 1M steps): pure-profit reward, AcceptFlag action space,
# all 6 contexts, one array task per seed (seed = array task id).
#
#   sbatch slurm/train_scml_1m.sh
#   STEPS=3000000 sbatch --time=12:00:00 slurm/train_scml_1m.sh   # 3M variant
#
# Wall-time math: with 64 CPUs and MAX_PARALLEL_MODELS=6 all 6 contexts train
# in ONE batch (6 learners x 8 envs ~ 54 busy cores); ~110 it/s -> 1M steps
# ~2.5h + eval pauses -> ~3h/task; 6h limit leaves margin. (On a 48-CPU
# allocation drop MAX_PARALLEL_MODELS to 3 and double the time limit.) Long-run safety nets (train.py):
#   - step-tagged checkpoints every CHECKPOINT_FREQ steps (_ckpt<steps>.zip)
#   - best-eval model saved to _best.zip (+_best_meta.json) at every record
#   - RESUME=1: a resubmitted task continues from its newest checkpoint, so a
#     timeout costs at most CHECKPOINT_FREQ steps. Resume a dead seed with:
#       sbatch --array=<seed> slurm/train_scml_1m.sh
# Models -> myagent/models/mymodel<ctx>_<RUN_NAME>_s<seed>_seed<seed>.zip
# Curves -> log/tensorboard_logs/<RUN_NAME>_s<seed>/. The 400k checkpoint of
# this run is directly comparable to baseline/400k_steps/ (same eval worlds).
set -euo pipefail
module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"
cd "$SLURM_SUBMIT_DIR"

# Refuse to launch a long run on a checkout that lacks the checkpoint/resume
# safety net (e.g. the cluster clone wasn't pulled) — a timeout would then
# lose everything.
if ! grep -q "CHECKPOINT_FREQ" myagent/train.py; then
    echo "FATAL: myagent/train.py has no checkpoint support — pull the latest commit first." >&2
    exit 1
fi

export SEED=$SLURM_ARRAY_TASK_ID
export RUN_NAME="${RUN_NAME:-long1m}_s${SEED}"
STEPS="${STEPS:-1000000}"

# Eval every 100k (not 40k): evals run serially inside each training process
# (~2-3 min per point at N_EVAL_EPISODES=10), and 100k still lands an eval on
# the 400k mark for comparison with the 400k baseline curves.
export EVAL_FREQ="${EVAL_FREQ:-100000}"
export N_EVAL_EPISODES="${N_EVAL_EPISODES:-10}"

# Long-run hygiene: checkpoints + resume on, per-step reward CSVs off (multi-GB
# at 1M+ steps), progress bars off (tens of MB of tqdm in the .err logs).
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-100000}"
# All 6 contexts in one batch (needs the 64-CPU allocation above).
export MAX_PARALLEL_MODELS="${MAX_PARALLEL_MODELS:-6}"
export RESUME=1
export LOG_REWARD_COMPONENTS=0
export PROGRESS_BARS=0

# AcceptFlag space is the in-code default since 2026-07-07; exported explicitly
# so this script stays correct even on an older checkout.
export ACTION_MANAGER="${ACTION_MANAGER:-accept}"
# TRAIN_CONTEXTS unset => all 6 contexts (override to train a subset).

# --- REWARD: PURE-PROFIT ------------------------------------------------------
export REWARD_SCORE_DELTA_WEIGHT=100.0
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
echo "SEED=${SEED}  RUN_NAME=${RUN_NAME}  STEPS=${STEPS}"
echo "TRAIN_CONTEXTS=${TRAIN_CONTEXTS:-all}  EVAL_FREQ=${EVAL_FREQ}  N_EVAL_EPISODES=${N_EVAL_EPISODES}"
echo "CHECKPOINT_FREQ=${CHECKPOINT_FREQ}  RESUME=${RESUME}  ACTION_MANAGER=${ACTION_MANAGER}  MAX_PARALLEL_MODELS=${MAX_PARALLEL_MODELS}"
echo "REWARD=pure-profit  REWARD_SCORE_DELTA_WEIGHT=${REWARD_SCORE_DELTA_WEIGHT}  (all shaping weights = 0)"

echo "=== Start training (seed ${SEED}, ${STEPS} steps) ==="
python -m myagent.train ${STEPS}
echo "=== Done seed ${SEED} ==="
date
