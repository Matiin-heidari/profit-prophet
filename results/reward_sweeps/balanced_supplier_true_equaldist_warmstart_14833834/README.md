# BalancedSupplier true EqualDist BC warmstart, job 14833834

Experiment:
- Context: BalancedSupplierContext
- Warm start model: behavior-cloned true EqualDistOneShotAgent
- PPO fine-tuning: 400000 steps
- Seeds: 0, 1, 2
- Eval frequency: 40000
- Eval episodes: 10
- Warm start model directory used: myagent/models/bc_equaldist_true_bs_test

Main result:
- mean best_gap_vs_best: about -0.198
- mean last_gap_vs_best: about -0.220
- mean best_my_score: about 0.955
- mean last_my_score: about 0.933
- mean best_rank: about 7.7
- mean last_rank: about 8.03

Interpretation:
The true EqualDist behavior-cloning warmstart is much better than the earlier failed bc_equal warmstart, but PPO fine-tuning does not monotonically improve it. The best checkpoint often occurs before the final step.
