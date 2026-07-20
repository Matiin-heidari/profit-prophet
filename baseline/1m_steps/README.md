# Baseline snapshot: 1M-step AcceptFlag deployment (2026-07-15)

The canonical agent as deployed on 2026-07-15, replacing
`baseline/400k_steps/`. **The first model set validated end-to-end on the
qualifier pool** rather than on the default-pool eval alone.

Training protocol: **1M steps** (resumed from checkpoints across two jobs),
pure-profit `REWARD_SCORE_DELTA_WEIGHT=100`, PPO `ent_coef=0.01,
gamma=0.95, n_steps=512, batch_size=256`, `ACTION_MANAGER=accept`,
per-context best of seeds {0,1,2} (runs `long1m_s{0,1,2}`, SLURM
14723915 → 14733957). Sources are the `_best.zip` best-eval snapshots, so
a context's file may predate its final step.

## models/ — best seed per context, `0_key/score` (10 fixed eval worlds)

| context | source seed | score | @ step |
|---|---|---|---|
| StrongSupplier   | long1m_s1 | 0.9954 | 1.0M |
| BalancedSupplier | long1m_s0 | 1.0308 | 1.0M |
| WeakSupplier     | long1m_s1 | 1.0522 | 800k |
| StrongConsumer   | long1m_s0 | 1.0501 | 1.0M |
| BalancedConsumer | long1m_s0 | 1.0438 | 1.0M |
| WeakConsumer     | long1m_s1 | 1.0500 | 1.0M |

Beats the 400k set in all six contexts (+0.016 to +0.049 on the mean over
seeds), and the StrongSupplier regression that dogged the 400k campaign is
gone.

## Qualifier-pool validation (the reason this set is deployed)

Paired SHARD_SEED benchmark vs the 400k set, 5 shards / 445 worlds per
agent, identical world configs (`bench_{current,long1m}_0713_2358`):

| set | pooled score | rank | gap_to_best |
|---|---|---|---|
| 400k (`current`) | 1.0143 | 9/11 | −0.111 |
| **1M (`long1m`)** | **1.0572** | **8/11** | **−0.070** |

5/5 paired shards in favor (mean +0.048). First time our agent overtakes a
qualifier in the pooled ranking (RTAgent, 1.0362).

A 3M continuation was also trained and benchmarked (`candidate_models/long3m/`,
`bench_long3m_0713_2358`) and regressed to 1.0369 / rank 9

## tensorboard_logs/ — `long1m_s{0,1,2}`

These curves extend past 1M because the 3M continuation reused the
same `RUN_NAME`. The deployed models correspond to the ≤1M portion (see the
per-context steps in the table above). Duplicate eval points at 100k
boundaries are resume double-logging.
