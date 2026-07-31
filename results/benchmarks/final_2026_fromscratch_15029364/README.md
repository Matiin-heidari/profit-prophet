# Final 2026 benchmark: PPO from scratch

Setup:
- Benchmark year: 2026
- Model set: PPO models trained from scratch
- Warm start: disabled
- Contexts: all six context-specific models
- n-configs: 10
- n-steps: 50
- Execution: serial
- Source run: final_2026_15029364

Result:
- MyAgent rank: 17 / 17
- MyAgent score: 0.9717
- Best score: 1.1225
- Gap to best: -0.1508
- MyAgent worlds: 320
- Saved scores: scores.csv

Notes:
This benchmark is the from-scratch comparison run. It was produced before the
warmstart models were copied into the final model set. It can be used as the
baseline for comparing the BalancedSupplier-only hybrid warmstart and the
all-context warmstart benchmark.
