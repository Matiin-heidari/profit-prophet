#!/bin/bash
#SBATCH --job-name=scml_pbrsday
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --partition=kisski
#SBATCH --array=0-2

# PBRS day-profit A/B arm: pure-profit + potential-based shaping
# with Φ = realized day profit (REWARD_POTENTIAL_KIND=dayprofit). Policy-
# invariant by construction; the A/B tests whether densifying the ALIGNED
# per-deal credit speeds learning where Φ=coverage (which densified the
# misaligned coverage signal) made it worse.
#
#   sbatch slurm/train_scml_pbrsday.sh
#
# BASELINE ARM = the completed accept_s{0,1,2} runs: identical protocol
# (400k steps, accept space, pure-profit w=100, new PPO, EVAL_FREQ=40k,
# N_EVAL_EPISODES=10, seeds 0-2), so no baseline re-run is needed. ONLY the
# potential weight/kind differ. Judge by paired final + matched-step
# 0_key/score per (context, seed). Kill criterion: a majority of paired
# losses ends PBRS for good (two failed potentials = idea falsified here).
set -euo pipefail
module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/envs/agentic"
cd "$SLURM_SUBMIT_DIR"

if ! grep -q "REWARD_POTENTIAL_KIND" myagent/train.py; then
    echo "FATAL: myagent/train.py has no dayprofit potential — pull the latest commit first." >&2
    exit 1
fi

export SEED=$SLURM_ARRAY_TASK_ID
export RUN_NAME="${RUN_NAME:-pbrsday}_s${SEED}"
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

# --- REWARD: PURE-PROFIT + DAY-PROFIT PBRS ------------------------------------
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
# THE experimental variables. Weight from the local scale probe
# (scripts/measure_reward_scale.py, 1500 random steps/context, 2026-07-07):
# at weight 1.0 pbrs mean|.| was 2-8x score_delta_bonus across contexts —
# the same "shaping dominates" regime that hurt Φ=coverage. 0.1 puts pbrs at
# ~0.2-0.8x score_delta (below target 0.3-1x only in the two contexts whose
# score_delta is already largest, i.e. where densification matters least).
export REWARD_POTENTIAL_WEIGHT="${REWARD_POTENTIAL_WEIGHT:-0.1}"
export REWARD_POTENTIAL_KIND=dayprofit

echo "=== SEED=${SEED}  (array task ${SLURM_ARRAY_TASK_ID}) ==="
echo "=== Host ==="; hostname
echo "=== Date ==="; date
echo "=== Python ==="; which python; python --version
echo "=== Git ==="; git rev-parse --short HEAD; git branch --show-current

echo "=== Run config ==="
echo "SEED=${SEED}  RUN_NAME=${RUN_NAME}  STEPS=${STEPS}"
echo "TRAIN_CONTEXTS=${TRAIN_CONTEXTS:-all}  EVAL_FREQ=${EVAL_FREQ}  N_EVAL_EPISODES=${N_EVAL_EPISODES}"
echo "ACTION_MANAGER=${ACTION_MANAGER}  MAX_PARALLEL_MODELS=${MAX_PARALLEL_MODELS}"
echo "REWARD=pure-profit + PBRS  POTENTIAL_WEIGHT=${REWARD_POTENTIAL_WEIGHT}  POTENTIAL_KIND=${REWARD_POTENTIAL_KIND}"

echo "=== Start training (seed ${SEED}, ${STEPS} steps) ==="
python -m myagent.train ${STEPS}
echo "=== Done seed ${SEED} ==="
date
