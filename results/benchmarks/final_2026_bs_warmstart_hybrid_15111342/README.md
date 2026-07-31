# Final 2026 benchmark: BalancedSupplier warmstart hybrid

Setup:
- Benchmark year: 2026
- Model set: hybrid
- BalancedSupplierContext: PPO fine-tuned from EqualDist BC warmstart
- Other contexts: previous final_all_contexts models
- n-configs: 10
- n-steps: 50
- serial execution

Result:
- MyAgent rank: 16 / 17
- MyAgent score: 0.9803
- Best score: 1.0974
- Gap to best: -0.1171
- Worlds for MyAgent: 320

Output:
- scores.csv contains the saved agent-world scores from the benchmark.
