# ProfitProphets

ProfitProphets is a group project developing an autonomous agent for the **Supply Chain Management League (SCML)**.

## Requirements

- Python **3.11**
- Recommended: Linux environment (some dependencies may not have full Windows support)

## Viewing Training Logs

Training metrics are logged to `./tensorboard_logs/<context_name>/` via TensorBoard.

To view them, run:

```bash
tensorboard --logdir ./tensorboard_logs
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

Metrics are split into two groups:

- `agent/` — per-agent stats for the RL agent (score, shortfall, deal rate, rank, etc.)
- `world/` — world-level stats averaged across eval episodes (welfare, negotiation counts, etc.)

Evaluation runs every 10% of training steps, so a full training run produces ~10 data points per metric.