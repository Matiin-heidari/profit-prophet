# Baseline snapshot: 400k-step AcceptFlag deployment (2026-07-07)

The canonical agent as deployed on 2026-07-07.

Training protocol: 400k steps, pure-profit `REWARD_SCORE_DELTA_WEIGHT=100`,
PPO `ent_coef=0.01, gamma=0.95, n_steps=512, batch_size=256`,
`ACTION_MANAGER=accept`, per-context best of seeds {0,1,2}
(runs `accept_s{0,1,2}`, SLURM 14704283 / 14706321 / 14720893).

## models/ — best seed per context, final `0_key/score` (10 fixed eval worlds)

| context | source seed | score |
|---|---|---|
| StrongSupplier   | accept_s2 | 0.9769 |
| BalancedSupplier | accept_s1 | 0.9840 |
| WeakSupplier     | accept_s1 | 1.0178 |
| StrongConsumer   | accept_s0 | 0.9932 |
| BalancedConsumer | accept_s1 | 1.0215 |
| WeakConsumer     | accept_s2 | 1.0219 |

