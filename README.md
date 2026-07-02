# ProfitProphets

ProfitProphets is a group project developing an autonomous agent for the **Supply Chain Management League (SCML)**.

## Requirements

- Python **3.11**
- Recommended: Linux environment (some dependencies may not have full Windows support)

## Running Training

```bash
python -m myagent.train [steps]
```

`steps` is the number of training timesteps per context (default **300000**). One
model is trained per context in `TRAIN_CONTEXTS`.

## Configuration (Environment Variables)

Training is configured through environment variables (no code changes needed).
Set them inline, e.g.:

```bash
TRAIN_CONTEXTS=WeakSupplierContext RUN_NAME=experiment1 LOG_REWARD_COMPONENTS=1 \
  python -m myagent.train 150000
```

### Training control

| Variable | Default | Description |
| --- | --- | --- |
| `TRAIN_CONTEXTS` | all 6 contexts | Comma-separated contexts to train. Valid names: `StrongSupplierContext`, `BalancedSupplierContext`, `WeakSupplierContext`, `StrongConsumerContext`, `BalancedConsumerContext`, `WeakConsumerContext`. |
| `RUN_NAME` | `default` | Label for output folders (`log/tensorboard_logs/<RUN_NAME>/`, `log/reward_component_logs/<RUN_NAME>/`). Use a unique name per run so logs don't overwrite each other. |
| `EVAL_FREQ` | `steps / 5` | Timesteps between evaluations. |
| `N_EVAL_EPISODES` | `3` | Number of evaluation worlds run per evaluation. |
| `DIAGNOSTICS_FREQ` | `steps / 20` | Timesteps between training-diagnostics logging (observation/action/reward summaries). |
| `RL_AGENT_CODE` | `On` | Short code used to identify our RL agent in `world.scores()` during evaluation. |

### Reward shaping weights

The reward is a sum of weighted terms. Each weight is resolved with the
precedence: **environment variable → per-context default → global default**.
So an env var here *overrides* the per-context defaults baked into
`_CONTEXT_DEFAULT_WEIGHTS` for **every** context in the run.

| Variable | Global default | Description |
| --- | --- | --- |
| `REWARD_SCORE_DELTA_WEIGHT` | `3.0` | Weight on the change in the agent's score (`balance / initial_balance`) this step — the actual profit signal. |
| `REWARD_NEED_WEIGHT` | `0.0` | Penalty for unmet need (quantity still to be traded), normalized by capacity and scaled by time pressure. |
| `REWARD_SHORTFALL_WEIGHT` | `0.0` | Penalty = unmet need × the world's actual shortfall penalty × time pressure. |
| `REWARD_OVERSHOOT_WEIGHT` | `0.0` | Penalty for overshoot (committing beyond need). |
| `REWARD_DISPOSAL_WEIGHT` | `0.0` | Penalty = overshoot × disposal cost. |
| `REWARD_PRODUCTIVITY_WEIGHT` | `0.0` | Bonus proportional to the fraction of need covered. |
| `REWARD_TIME_PRESSURE_WEIGHT` | `1.0` | Controls how strongly need/shortfall penalties grow as the day progresses (`time_multiplier = 1 + weight × relative_time`). |
| `REWARD_NEED_NORMALIZER` | `0.0` | If `> 0`, divide need by this constant instead of by `n_lines` (production capacity). |
| `REWARD_PRICE_WEIGHT` | `0.0` | Bonus for closing deals that beat the catalog price (above when selling, below when buying), quantity-weighted. |
| `REWARD_DEAL_WEIGHT` | `0.0` | Bonus proportional to realized trade volume this step (normalized by capacity). |
| `REWARD_ENGAGEMENT_WEIGHT` | `0.0` | Bonus for keeping negotiations alive (not ending them early). |

Note: the global defaults above apply when neither an env var nor a per-context
default is set.

### Per-context defaults

When the corresponding `REWARD_*` env var is **not** set, each context falls back
to these defaults (from `_CONTEXT_DEFAULT_WEIGHTS` in `myagent/train.py`). Any
weight not listed uses the global default from the table above (notably
`score_delta_weight = 3.0` and `time_pressure_weight = 1.0`, which apply to every
context). The Supplier and Consumer variants of each strength share the same
weights — the buy/sell side is auto-detected from the context, so the same
"good price" logic works on either side.

| Context (Supplier & Consumer) | `price_weight` | `deal_weight` | `need_weight` | Intent |
| --- | --- | --- | --- | --- |
| `Strong*` | `0.50` | – | `0.05` | Price is the primary driver (strong pricing leverage); small `need` gives a dense gradient toward trading at all. |
| `Balanced*` | `0.30` | – | `0.05` | Moderate price emphasis plus light coverage pressure. |
| `Weak*` | – | `0.20` | `0.05` | Realized trade volume is the main signal (demand is scarce); small `need` keeps coverage pressure without dominating. |

All six contexts additionally inherit `score_delta_weight = 3.0` (profit signal)
and `time_pressure_weight = 1.0`; the remaining weights default to `0.0`.

### Logging & diagnostics

| Variable | Default | Description |
| --- | --- | --- |
| `LOG_REWARD_COMPONENTS` | `0` | `1` writes a per-step CSV of every reward component to `log/reward_component_logs/<RUN_NAME>/<context>/<job_id>/`. Useful for debugging the reward; adds I/O overhead, so keep off for production runs. |
| `REWARD_LOG_FLUSH_EVERY` | `1000` | Flush the reward-component CSV every N rows (only relevant when `LOG_REWARD_COMPONENTS=1`). |
| `LOG_WORLD` | `0` | `1` enables a verbose, fail-fast world (debug mode, saves contracts/negotiations, exceptions propagate). Default `0` is the robust profile (agent/negotiation exceptions ignored, minimal logging) — use that for long unattended runs. |

### System (set automatically by SLURM)

These are read if present but normally set by the scheduler, not by hand:

| Variable | Description |
| --- | --- |
| `SLURM_CPUS_PER_TASK` | Number of CPUs available; sizes the parallel environment/model counts. |
| `SLURM_JOB_ID` | Included in reward-component log paths to separate jobs. |

## Viewing Training Logs

Training metrics are logged to `./log/tensorboard_logs/<RUN_NAME>/<context_name>/` via TensorBoard.

To view them, run:

```bash
tensorboard --logdir ./log/tensorboard_logs
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

Metrics are split into two groups:

- `agent/` — per-agent stats for the RL agent (score, shortfall, deal rate, rank, etc.)
- `world/` — world-level stats averaged across eval episodes (welfare, negotiation counts, etc.)

By default evaluation runs every 20% of training steps (`EVAL_FREQ = steps / 5`), so a full training run produces ~5 data points per metric. Adjust with `EVAL_FREQ` / `N_EVAL_EPISODES`.