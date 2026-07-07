#!/bin/bash
#SBATCH --job-name=scml_pbrs
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=kisski

module load miniforge3/24.3.0-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $PROJECT_DIR/envs/agentic

cd "$SLURM_SUBMIT_DIR"

# --- PURE PROFIT + POTENTIAL-BASED SHAPING (PBRS) --------------------------
# Identical to the pure-profit control, but adds a policy-invariant potential
# term (gamma*Phi(s') - Phi(s), Phi = coverage of active need). PBRS densifies
# the sparse profit signal WITHOUT changing the optimal policy (Ng et al. 1999),
# so this run is designed to show improved SAMPLE EFFICIENCY over pure profit:
# the same/better score reached in fewer steps. Compare its 0_key/* learning
# curves directly against pure_profit_400k.
export REWARD_SCORE_DELTA_WEIGHT=100.0
export REWARD_POTENTIAL_WEIGHT=0.5     # the only difference vs pure-profit
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

export TRAIN_CONTEXTS="${TRAIN_CONTEXTS:-StrongSupplierContext,WeakSupplierContext}"
export RUN_NAME="${RUN_NAME:-pbrs_400k}"
export EVAL_FREQ="${EVAL_FREQ:-40000}"
export N_EVAL_EPISODES="${N_EVAL_EPISODES:-10}"
export LOG_REWARD_COMPONENTS=1

echo "=== Host ==="; hostname
echo "=== Date ==="; date
echo "=== Python ==="; which python; python --version
echo "=== Git ==="; git rev-parse --short HEAD; git branch --show-current
echo "=== Run config (pure profit + PBRS) ==="
echo "RUN_NAME=${RUN_NAME}"
echo "TRAIN_CONTEXTS=${TRAIN_CONTEXTS}"
echo "EVAL_FREQ=${EVAL_FREQ}  N_EVAL_EPISODES=${N_EVAL_EPISODES}"
echo "REWARD_SCORE_DELTA_WEIGHT=${REWARD_SCORE_DELTA_WEIGHT}  REWARD_POTENTIAL_WEIGHT=${REWARD_POTENTIAL_WEIGHT}  (all other shaping = 0)"

echo "=== Start training (400k, pure profit + PBRS) ==="
# Pin the pre-2026-07-07 action space: this script reproduces arms that were
# defined on FlexibleActionManager (the in-code default is now "accept").
export ACTION_MANAGER="${ACTION_MANAGER:-flexible}"
python -m myagent.train 400000
echo "=== Done ==="; date
