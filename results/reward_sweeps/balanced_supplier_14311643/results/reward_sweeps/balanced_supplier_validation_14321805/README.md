# Reward validation sweep: BalancedSupplierContext

Validation job: 14321805  
Context: BalancedSupplierContext  
Branch: reward-function  

This folder contains:
- slurm/: Slurm output/error logs for validation array jobs 0-17
- scripts/: Validation sweep script used for the run
- summaries/sacct_14321805.txt: Slurm completion status
- summaries/validation_summary.csv: Aggregated metrics per reward configuration
- summaries/validation_individual_runs.csv: Metrics for each individual repeated run

Validation setup:
- 6 reward configurations
- 3 repeats per configuration
- 18 Slurm array tasks
- N_EVAL_EPISODES=5
- LOG_REWARD_COMPONENTS=0

Main ranking criterion:
eval/score_gap_vs_best_opponent

Main conclusion:
The validation did not show a clear win for need=0.0001 + shortfall=0.0001 over baseline.
The next code step should be best-checkpoint saving rather than another reward-weight sweep.
