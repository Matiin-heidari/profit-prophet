# Final 2026 benchmark: all-context warmstart

Setup:
- Benchmark year: 2026
- Model set: all-context warmstart
- All six context models were PPO fine-tuned from EqualDist BC warmstarts
- n-configs: 10
- n-steps: 50
- Execution: serial
- Model directory used: myagent/models_all_warmstart_final

Result:
- MyAgent rank: 16 / 17
- MyAgent score: 0.9904
- Best score: 1.1009
- Gap to best: -0.1104
- MyAgent worlds: 320
- Saved scores: scores.csv

Context usage:
- StrongConsumerContext: 96 worlds
- BalancedConsumerContext: 80 worlds
- BalancedSupplierContext: 80 worlds
- StrongSupplierContext: 32 worlds
- WeakSupplierContext: 32 worlds
- fallback: 0 worlds

Notes:
This is the final all-context warmstart benchmark. Compared with the
BalancedSupplier-only hybrid benchmark, the all-context warmstart improved the
mean score from 0.9803 to 0.9904 and reduced the gap to best from -0.1171 to
-0.1104, but the rank remained 16 / 17.
