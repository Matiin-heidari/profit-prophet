#!/bin/bash
#SBATCH --job-name=scml_oppool
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --partition=kisski
#SBATCH --array=0-2

# Strong-opponent-pool A/B arm: OPPONENT_POOL=strong mixes the top-2024
# qualifiers (Cautious, Suzuka, DistRedist — see myagent/opponents.py) into
# the TRAINING worlds' non-competitor slots, so the training distribution
# matches what the qualifier benchmark measures. Eval worlds keep the default
# pool (comparable 0_key/* curves); deployment is untouched.
#
#   sbatch slurm/train_scml_oppool.sh
#
# BASELINE ARM = the completed accept_s{0,1,2} runs (identical protocol; ONLY
# OPPONENT_POOL differs). This item is FOR transfer: a paired qualifier
# benchmark (same SHARD_SEED across both model sets) is the primary judge —
# the saturated default-pool eval may legitimately not move.
#
# Wall time: local smoke probe (3000 random steps, BalancedSupplier) measured
# ~parity between pools (122 steps/s strong vs 106 default — the curated
# qualifiers are cheap heuristics), so expect ~the baseline's 1.5-2h; the 6h
# limit is margin. CHECKPOINT_FREQ+RESUME=1 make a timeout cheap anyway
# (resubmit the seed with sbatch --array=<seed>).
set -euo pipefail
module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"
cd "$SLURM_SUBMIT_DIR"

if [ ! -f myagent/opponents.py ]; then
    echo "FATAL: myagent/opponents.py missing — pull the latest commit first." >&2
    exit 1
fi

export SEED=$SLURM_ARRAY_TASK_ID
export RUN_NAME="${RUN_NAME:-oppool}_s${SEED}"
STEPS="${STEPS:-400000}"

# Match the accept_s* baseline protocol exactly.
export EVAL_FREQ="${EVAL_FREQ:-40000}"
export N_EVAL_EPISODES="${N_EVAL_EPISODES:-10}"

export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-100000}"
export MAX_PARALLEL_MODELS="${MAX_PARALLEL_MODELS:-6}"
export RESUME=1
export LOG_REWARD_COMPONENTS="${LOG_REWARD_COMPONENTS:-0}"
export PROGRESS_BARS=0
export ACTION_MANAGER="${ACTION_MANAGER:-accept}"

# THE experimental variable.
export OPPONENT_POOL="${OPPONENT_POOL:-strong}"

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
echo "ACTION_MANAGER=${ACTION_MANAGER}  MAX_PARALLEL_MODELS=${MAX_PARALLEL_MODELS}"
echo "OPPONENT_POOL=${OPPONENT_POOL}  REWARD=pure-profit"

echo "=== Start training (seed ${SEED}, ${STEPS} steps) ==="
python -m myagent.train ${STEPS}
echo "=== Done seed ${SEED} ==="
date
