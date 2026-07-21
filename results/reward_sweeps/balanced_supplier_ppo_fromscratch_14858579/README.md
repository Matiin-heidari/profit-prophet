# BalancedSupplier PPO from scratch, job 14858579

Setup:
- Context: BalancedSupplierContext
- PPO training: from scratch
- Training steps: 400000
- Seeds: 0, 1, 2
- Eval frequency: 40000
- Eval episodes: 10
- Warm start: disabled

Summary:
- mean best_gap_vs_best: -0.339
- mean last_gap_vs_best: -0.417
- mean best_my_score: 0.822
- mean last_my_score: 0.747
- mean best_rank: 9.2
- mean last_rank: 9.5

Notes:
This run is the control group for the EqualDist behavior-cloning warmstart
experiment in BalancedSupplierContext.
