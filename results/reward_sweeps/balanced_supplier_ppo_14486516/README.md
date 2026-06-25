# PPO reward sweep: BalancedSupplierContext

Sweep job: 14486516  
Context: BalancedSupplierContext  
Branch: reward-function  

This folder contains:
- slurm/: Slurm output/error logs for PPO array jobs 0-15
- scripts/: PPO sweep script used for the run
- summaries/sacct_14486516.txt: Slurm completion status
- summaries/ppo_sweep_summary.csv: PPO-only TensorBoard evaluation summary
- summaries/a2c_vs_ppo_sweep_comparison.csv: Comparison against the earlier A2C sweep

Comparison baseline:
results/reward_sweeps/balanced_supplier_14311643

Main ranking criterion:
eval/score_gap_vs_best_opponent

Main observation:
The best PPO variant was ppo_sweep_bs_08_need_00025_shortfall_0001.
It was close to but slightly below the best A2C variant sweep_bs_07_need_0001_shortfall_0001 in this single-run sweep.
