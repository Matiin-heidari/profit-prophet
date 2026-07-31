# Final 2026 benchmark: BalancedSupplier warmstart hybrid

Setup:
- Benchmark year: 2026
- Model set: hybrid
- BalancedSupplierContext: PPO fine-tuned from EqualDist BC warmstart
- Other contexts: previous final models
- n-configs: 10
- n-steps: 50
- Execution: serial

Result:
- MyAgent rank: 16 / 17
- MyAgent score: 0.9803
- Best score: 1.0974
- Gap to best: -0.1171
- MyAgent worlds: 320
- Saved scores: scores.csv

Context usage:
- StrongSupplierContext: 96 worlds
- BalancedSupplierContext: 64 worlds
- BalancedConsumerContext: 64 worlds
- WeakSupplierContext: 48 worlds
- StrongConsumerContext: 48 worlds
- fallback: 0 worlds

Notes:
This benchmark only replaces the BalancedSupplierContext model with the
warmstart-finetuned model. It is therefore a hybrid comparison, not the final
all-context warmstart result.
