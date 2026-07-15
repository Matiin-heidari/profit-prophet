# BalancedSupplier PPO from scratch, job 14858579

Experiment:
- Context: BalancedSupplierContext
- PPO training from scratch
- Seeds: 0, 1, 2
- Training steps: 400000
- Eval frequency: 40000
- Eval episodes: 10
- Warm start: disabled

Purpose:
Direct control group for the true EqualDist behavior-cloning warmstart run 14833834.

Main result:
- mean best_gap_vs_best: about -0.339
- mean last_gap_vs_best: about -0.417
- mean best_my_score: about 0.822
- mean last_my_score: about 0.747
- mean best_rank: about 9.2
- mean last_rank: about 9.5

Comparison:
The true EqualDist BC warmstart run 14833834 performs substantially better:
- best_gap_vs_best improves from about -0.339 to about -0.198
- last_gap_vs_best improves from about -0.417 to about -0.220
- best_my_score improves from about 0.822 to about 0.955
- last_my_score improves from about 0.747 to about 0.933
- best rank improves from about 9.2 to about 7.7
- final rank improves from about 9.5 to about 8.03

Interpretation:
PPO from scratch learns a weaker policy under the same setup. The behavior-cloned true EqualDist warmstart gives a clear improvement, although the warmstarted model still does not match true EqualDist performance and may degrade during PPO fine-tuning.
