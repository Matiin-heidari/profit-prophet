# BalancedSupplier true EqualDist BC warmstart, job 14833834

Setup:
- Context: BalancedSupplierContext
- Warm start model: behavior-cloned EqualDistOneShotAgent
- PPO fine-tuning: 400000 steps
- Seeds: 0, 1, 2
- Eval frequency: 40000
- Eval episodes: 10
- Warm start directory: myagent/models/bc_equaldist_true_bs_test

Summary:
- mean best_gap_vs_best: -0.198
- mean last_gap_vs_best: -0.220
- mean best_my_score: 0.955
- mean last_my_score: 0.933
- mean best_rank: 7.7
- mean last_rank: 8.03

Notes:
The EqualDist behavior-cloning warmstart improved PPO performance in this
BalancedSupplier setup. The best evaluation point often occurred before the
final training step, so final checkpoints should not be treated as necessarily
optimal.
