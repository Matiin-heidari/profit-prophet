# Reward sweep: BalancedSupplierContext

Sweep job: 14311643  
Context: BalancedSupplierContext  
Branch: reward-function  

This folder contains:
- slurm/: Slurm output/error logs for array jobs 0-15
- scripts/: Sweep script used for the run
- summaries/sacct_14311643.txt: Slurm completion status
- summaries/sweep_summary.csv: TensorBoard evaluation summary
- summaries/reward_component_summary.csv: Aggregated reward component statistics

Main ranking criterion:
eval/score_gap_vs_best_opponent

Best candidate in this sweep:
sweep_bs_07_need_0001_shortfall_0001
