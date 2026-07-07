# Candidate model sets

Best-seed-per-context model sets for benchmarking, selected 2026-07-07 from the
two full 6-context training campaigns (see CLAUDE.md §4 "Full 6-context runs").
Selection criterion: final `0_key/score` (mean over the 10 fixed eval worlds)
at 400k steps.

Benchmark a set WITHOUT moving any files via the `MODEL_DIR` env var
(read in `myagent/common.py`; relative paths resolve from the repo root):

    MODEL_DIR=candidate_models/flex   RUN_NAME=bench_flex   sbatch slurm/benchmark_shard.sh
    MODEL_DIR=candidate_models/accept RUN_NAME=bench_accept sbatch slurm/benchmark_shard.sh

or submit both at once with `slurm/benchmark_candidates.sh`. `MyAgent`
auto-detects the action manager per model from its saved action space, so the
two sets need no further configuration.

## flex/ — Updated PPO + weight (FlexibleActionManager)

| context | source seed | final 0_key/score |
|---|---|---|
| StrongSupplier   | seeds_s0 | 0.9436 ⚠️ (below the old-baseline 0.982 — known regression, CLAUDE.md §4) |
| BalancedSupplier | seeds_s0 | 0.9546 |
| WeakSupplier     | seeds_s2 | 1.0135 |
| StrongConsumer   | seeds_s1 | 1.0168 |
| BalancedConsumer | seeds_s2 | 1.0208 |
| WeakConsumer     | seeds_s2 | 1.0331 |

## accept/ — New action manager (AcceptFlagActionManager)

| context | source seed | final 0_key/score |
|---|---|---|
| StrongSupplier   | accept_s2 | 0.9769 |
| BalancedSupplier | accept_s1 | 0.9840 |
| WeakSupplier     | accept_s1 | 1.0178 |
| StrongConsumer   | accept_s1 | 0.9915 |
| BalancedConsumer | accept_s1 | 1.0215 |
| WeakConsumer     | accept_s2 | 1.0219 |

Note: accept seed-0 Strong/Weak models exist (`old_models/0707 accept and
adjPPO/`) but were locally unevaluated (TB not synced) when this selection was
made; WeakConsumer accept_s0 has no model (timeout). Re-select if the rerun
produces better seeds.

Source files: `old_models/0707 accept and adjPPO/` (verbatim copies, renamed
to the canonical `mymodel<Context>.zip`).
