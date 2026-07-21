# ProfitProphets

ProfitProphets is a group project developing an autonomous agent for the **Supply Chain Management League (SCML)**.

## Requirements

- Python **3.11**
- Recommended: Linux environment (some dependencies may not have full Windows support)

## Installation

Create an environment and install the pinned requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Newer opponent pools (optional, for benchmarking)

The 2025/2026 competition agents are only in the `scml-agents` repository, not its
released package. Install them **into the same environment**, directly from GitHub:

```bash
pip install --no-deps git+https://github.com/yasserfarouk/scml-agents.git@29a3352
```

## Basic Usage

| Command | What it does |
| --- | --- |
| `python -m myagent.train [steps]` | Train the per-context models locally (default 300000 steps each). |
| `python -m myagent.myagent --year 2026` | Drop the agent into a quick test tournament against the 2026 pool. |
| `sbatch train_scml.sh` | Train all six contexts on the cluster (SLURM). |
| `./slurm/benchmark_candidates.sh` | Benchmark candidate model sets against the qualifier pool (SLURM). |

Each is covered in more detail in the sections below.

## Running Training

```bash
python -m myagent.train [steps]
```

`steps` is the number of training timesteps per context (default **300000**). One
model is trained per context in `TRAIN_CONTEXTS`.



## Benchmarking & Evaluation

Two entry points run `MyAgent` against real opponent pools. Deployment loads all
six per-context models automatically.

**`scripts/benchmark.py`** drops `MyAgent` into an ANAC-style tournament against a
year's qualifier pool and reports its rank, score, and gap to the best agent:

```bash
python scripts/benchmark.py --n-configs 10 --n-steps 50           # newest pool
python scripts/benchmark.py --year 2026 --n-configs 10            # 2026 pool (see below)
python scripts/benchmark.py --model-dir candidate_models/long1m   # benchmark an alternative model set
```

`--year 2026` (or `2025`) needs the optional `scml-agents` install from the
[Installation](#newer-opponent-pools-optional-for-benchmarking) section. Other
useful flags: `--save-scores` (per-world CSV for sharded runs), `--shard-seed` (pair the
world draw across model sets). At scale, run it sharded on the cluster — see
[Running on SLURM](#running-on-slurm-cluster).

**`python -m myagent.myagent`** is a lighter local harness for a quick check —
it drops `MyAgent` into a small test tournament:

```bash
python -m myagent.myagent                                              # default winners pool
python -m myagent.myagent --year 2026 --n-competitors-per-world 2
```

Pass `--year` to face a specific pool; set `--n-competitors-per-world 2` with
large pools, or the round-robin expands into thousands of worlds.

## Viewing Training Logs

Training metrics are logged to `./log/tensorboard_logs/<RUN_NAME>/<context_name>/` via TensorBoard.

To view them, run:

```bash
tensorboard --logdir ./log/tensorboard_logs
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

Metrics are split into groups:

- `0_key/` — the headline evaluation metrics, notably `0_key/my_rank`
- `agent/` — per-agent stats for the RL agent (score, shortfall, deal rate, rank, etc.)
- `world/` — world-level stats averaged across eval episodes (welfare, negotiation counts, etc.)

By default evaluation runs every 20% of training steps (`EVAL_FREQ = steps / 5`), so a full training run produces ~5 data points per metric. Adjust with `EVAL_FREQ` / `N_EVAL_EPISODES`.

## Running on SLURM (Cluster)

Training and benchmark jobs run on the cluster via `sbatch`. **Always submit from
the repo root**, e.g.
`sbatch slurm/train_scml_1m.sh`, and set a unique `RUN_NAME` per run so logs and
model files don't collide. The same environment variables from
[Configuration](#configuration-environment-variables) apply — pass them through
with `--export`, e.g. `sbatch --export=ALL,RUN_NAME=exp1,STEPS=1000000 …`.
This assumes the necessary environment is already installed. This should be the case if you're running on the projects' cluster.

| Script | What it does |
| --- | --- |
| `train_scml.sh` | Standard training run — all 6 contexts together (48 CPUs, 3 h). The simplest way to train one model set on the cluster. |
| `slurm/train_scml_1m.sh` | Flagship long run: 1M steps (`STEPS=3000000 sbatch --time=12:00:00 …` for 3M), all 6 contexts, one array task per seed (0–2), with checkpointing and `RESUME` support (64 CPUs, all contexts in parallel). |
| `slurm/benchmark_candidates.sh` | Submitter (run it directly, not via `sbatch`) that launches `benchmark_shard.sh` for one or more candidate model sets via `MODEL_DIR`, seeding the draw so the sets play identical worlds. |
| `run_scml_agent.sh` | Runs the deployed agent (all 6 models) in a quick tournament — a fast sanity check of a model set. |

After a sharded benchmark finishes, pool and re-rank the shards:

```bash
python scripts/aggregate_benchmark.py log/benchmark_shards/<RUN_NAME> --run <RUN_NAME>
```



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
| `CHECKPOINT_FREQ` | `100000` | Timesteps between step-tagged model checkpoints (`<model>_ckpt<steps>.zip`). `0` disables. Checkpoints let a timed-out long run resume and keep intermediate policies comparable across step budgets. The best-eval model is additionally saved to `<model>_best.zip` (+ `_best_meta.json` with score/step) whenever the fixed-world eval score sets a new record. |
| `MAX_PARALLEL_MODELS` | `3` | Cap on how many context models train simultaneously (each uses a learner + up to 8 env workers ≈ 9 busy cores). `3` fits a 48-CPU allocation (6 contexts = 2 batches); `6` on a 64-CPU node trains all contexts in one batch — half the wall time, identical training dynamics. |
| `PROGRESS_BARS` | `1` | `0` disables the tqdm progress bar. |
| `RESUME` | `0` | `1` = continue training from the newest `_ckpt<steps>.zip` toward the same total step count (`reset_num_timesteps=False`, TB curves continue; the best-model record is restored from the sidecar json). No checkpoint found = starts fresh. |
| `RL_AGENT_CODE` | `On` | Short code used to identify our RL agent in `world.scores()` during evaluation. |
| `MODEL_DIR` | `myagent/models` | Directory holding the 6 `mymodel<Context>.zip` files `MyAgent` loads (relative paths resolve from the CWD). Lets alternative model sets (e.g. `candidate_models/flex` vs `candidate_models/accept`) be benchmarked without moving files: `MODEL_DIR=candidate_models/flex sbatch slurm/benchmark_shard.sh`, or `--model-dir` on `scripts/benchmark.py`, or `slurm/benchmark_candidates.sh` to submit both sets at once. |
| `ACTION_MANAGER` | `accept` | Action space for training/eval. `accept`  = `AcceptFlagActionManager`, `flexible` = scml's `FlexibleActionManager` Models trained with different managers are **not** interchangeable; deployment (`MyAgent`) auto-detects the right manager per model from its saved action space. |
| `OPPONENT_POOL` | `default` | Non-competitor pool for **training** worlds (`myagent/opponents.py`). `default` = scml's Greedy/RandDist/EqualDist trio. `strong` = that trio **plus** the top-2024 qualifiers (Cautious, Suzuka, DistRedist). A comma-separated list of `module.path:ClassName` specs selects exactly those classes. |
| `SHARD_SEED` | unset | **Benchmark only** (`scripts/benchmark.py` / `slurm/benchmark_shard.sh`). Unset = each shard draws a random world config. `slurm/benchmark_candidates.sh` sets it automatically for the sets it submits. Pairs the config draw only. |

### Reward shaping weights

The reward is a sum of weighted terms. Each weight is resolved with the
precedence: **environment variable → per-context default → global default**.
So an env var here *overrides* the per-context defaults baked into
`_CONTEXT_DEFAULT_WEIGHTS` for **every** context in the run.

| Variable | Global default | Description |
| --- | --- | --- |
| `REWARD_SCORE_DELTA_WEIGHT` | `100.0` | Weight on the change in the agent's score (`balance / initial_balance`) this step — the actual profit signal. Kept large because PPO's critic fits raw (unnormalized) returns — too small a reward scale and it underfits, even though the policy loss itself is scale-insensitive (advantages are normalized per-batch). |
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
| `REWARD_MARGIN_WEIGHT` | `0.0` | Bonus from realized deals valued at break-even (`(price − break_even)·qty`, normalized) — a profit-aligned margin signal. |
| `REWARD_POTENTIAL_WEIGHT` | `0.0` | Potential-based reward shaping (PBRS): adds `γ·Φ(s') − Φ(s)` per step, which densifies a sparse signal without changing the optimal policy (Ng et al. 1999). `0` disables. Pick the weight so the pbrs term is ~0.3–1× the score_delta term (`scripts/measure_reward_scale.py`); a dominating potential hurts the learning path even though it cannot move the optimum. |
| `REWARD_POTENTIAL_KIND` | `coverage` | Which potential Φ the PBRS term uses (only matters when `REWARD_POTENTIAL_WEIGHT ≠ 0`). `coverage` = fraction of the active need already secured (legacy). `dayprofit` = realized profit of the current day so far, valued at break-even like the margin term.|

Note: the global defaults above apply when neither an env var nor a per-context
default is set.

### Logging & diagnostics

| Variable | Default | Description |
| --- | --- | --- |
| `LOG_REWARD_COMPONENTS` | `0` | `1` writes a per-step CSV of every reward component to `log/reward_component_logs/<RUN_NAME>/<context>/<job_id>/`. |
| `REWARD_LOG_FLUSH_EVERY` | `1000` | Flush the reward-component CSV every N rows (only relevant when `LOG_REWARD_COMPONENTS=1`). |
| `LOG_WORLD` | `0` | `1` enables a verbose, fail-fast world (debug mode, saves contracts/negotiations, exceptions propagate).|

### System (set automatically by SLURM)

These are read if present but normally set by the scheduler, not by hand:

| Variable | Description |
| --- | --- |
| `SLURM_CPUS_PER_TASK` | Number of CPUs available; sizes the parallel environment/model counts. |
| `SLURM_JOB_ID` | Included in reward-component log paths to separate jobs. |
