# ProfitProphets

ProfitProphets is an autonomous negotiation agent for the **Supply Chain Management League (SCML)**.

The agent learns its negotiation policy with **Proximal Policy Optimization (PPO)**. Its concession, acceptance and quantity-splitting behaviour is learned from experience instead of being defined entirely through hand-written negotiation rules.

## Approach

A single policy has to handle very different situations in SCML. An agent may act as a supplier or consumer and may have a strong, balanced or weak position in the market.

Instead of using one general policy for all situations, ProfitProphets uses six specialist policies:

| Role     | Market position |
| -------- | --------------- |
| Supplier | Strong          |
| Supplier | Balanced        |
| Supplier | Weak            |
| Consumer | Strong          |
| Consumer | Balanced        |
| Consumer | Weak            |

Training is performed in two stages:

1. A general model is pretrained to learn basic negotiation behaviour.
2. The pretrained model is fine-tuned separately for each of the six contexts.

During deployment, the agent determines its current market context from observable information and selects the corresponding policy.

```text
SCML environment
        |
        v
Market context
        |
        v
Selection of one specialist policy
        |
        v
PPO prediction
        |
        v
Negotiation action
```

The main reward signal is based on changes in the agent's normalized score. Additional reward components can be enabled for experiments with unmet need, shortfall, overshoot, deal volume, price and realized margin.

## Requirements

* Python 3.11
* Linux is recommended
* SLURM is optional and only required for the included cluster scripts

Some dependencies may not have full Windows support.

## Installation

Create a virtual environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional competition agents

The 2025 and 2026 competition agents are available in the `scml-agents` GitHub repository but are not included in its released Python package.

Install them into the same environment when benchmarking against these pools:

```bash
pip install --no-deps git+https://github.com/yasserfarouk/scml-agents.git@29a3352
```

This step is not required for normal training.

## Running the Final Agent

The deployed agent is implemented in `myagent/myagent.py`.

By default, it loads one policy for each of the six contexts from:

```text
myagent/models/
```

Run a small local tournament with:

```bash
python -m myagent.myagent
```

Run the agent against the 2026 competition pool with:

```bash
python -m myagent.myagent --year 2026 --n-competitors-per-world 2
```

Using two competitors per world is recommended for large opponent pools. Otherwise, the round-robin tournament may create several thousand worlds.

## Training

The main training command is:

```bash
python -m myagent.train [steps]
```

`steps` defines the number of training timesteps per context. The default is 300,000 timesteps.

Train all six policies with the default settings:

```bash
python -m myagent.train
```

Train all policies for one million timesteps each:

```bash
python -m myagent.train 1000000
```

The available training contexts are:

```text
StrongSupplierContext
BalancedSupplierContext
WeakSupplierContext
StrongConsumerContext
BalancedConsumerContext
WeakConsumerContext
```

Individual contexts can be selected through `TRAIN_CONTEXTS`:

```bash
TRAIN_CONTEXTS=WeakSupplierContext,BalancedSupplierContext \
RUN_NAME=selected_contexts \
python -m myagent.train 300000
```

A unique `RUN_NAME` should be used for every experiment so that model files and logs from different runs do not overwrite each other.

## Model Files

The default model directory is:

```text
myagent/models/
```

The six model files follow this naming scheme:

```text
mymodelStrongSupplierContext.zip
mymodelBalancedSupplierContext.zip
mymodelWeakSupplierContext.zip
mymodelStrongConsumerContext.zip
mymodelBalancedConsumerContext.zip
mymodelWeakConsumerContext.zip
```

Alternative model sets can be stored in separate directories and selected through `MODEL_DIR`.

Example:

```bash
MODEL_DIR=candidate_models/long1m \
python -m myagent.myagent --year 2026 --n-competitors-per-world 2
```

## Action Managers

Two action representations are supported:

| Value      | Action manager                 |
| ---------- | ------------------------------ |
| `accept`   | `AcceptFlagActionManager`      |
| `flexible` | SCML's `FlexibleActionManager` |

Select an action manager during training with:

```bash
ACTION_MANAGER=accept python -m myagent.train 300000
```

Models trained with different action managers are not interchangeable because their action spaces differ.

During deployment, `MyAgent` inspects the saved model and selects the matching action manager automatically.

## Benchmarking

The main benchmark script runs the agent in an ANAC-style tournament and reports its rank, score and gap to the best-performing agent.

Run a benchmark with the default opponent pool:

```bash
python scripts/benchmark.py --n-configs 10 --n-steps 50
```

Run against the 2026 competition pool:

```bash
python scripts/benchmark.py --year 2026 --n-configs 10
```

Benchmark an alternative model set:

```bash
python scripts/benchmark.py \
  --model-dir candidate_models/long1m \
  --year 2026 \
  --n-configs 10
```

Useful options include:

| Option          | Description                                       |
| --------------- | ------------------------------------------------- |
| `--year`        | Selects a competition opponent pool               |
| `--model-dir`   | Selects the directory containing the six policies |
| `--n-configs`   | Sets the number of tournament configurations      |
| `--n-steps`     | Sets the number of simulation steps               |
| `--save-scores` | Saves per-world scores to CSV                     |
| `--shard-seed`  | Fixes the world configuration draw                |

The 2025 and 2026 pools require the optional `scml-agents` installation described above.

### Comparing Candidate Models

Candidate model sets should be evaluated on the same world configurations.

The `--shard-seed` option fixes the configuration draw so that different model sets are tested under matching conditions.

```bash
python scripts/benchmark.py \
  --model-dir candidate_models/model_a \
  --year 2026 \
  --n-configs 10 \
  --shard-seed 42 \
  --save-scores
```

The command can then be repeated with another model directory while keeping the remaining arguments unchanged.

This reduces differences caused only by randomly generated worlds.

## Training Logs

Training metrics are stored in:

```text
log/tensorboard_logs/<RUN_NAME>/<context_name>/
```

Start TensorBoard with:

```bash
tensorboard --logdir ./log/tensorboard_logs
```

Then open:

```text
http://localhost:6006
```

The metrics are divided into three groups:

| Group    | Contents                                                       |
| -------- | -------------------------------------------------------------- |
| `0_key/` | Main evaluation metrics, including `0_key/my_rank`             |
| `agent/` | Score, shortfall, deal rate, rank and related agent statistics |
| `world/` | Welfare, negotiation counts and other world-level statistics   |

By default, evaluation is performed every 20% of the selected training duration:

```text
EVAL_FREQ = steps / 5
```

A complete training run therefore produces approximately five evaluation points per metric.

## Checkpoints

Step-tagged checkpoints are saved at regular intervals:

```text
mymodelStrongSupplierContext_ckpt100000.zip
```

The best model found during fixed-world evaluation is additionally stored as:

```text
mymodelStrongSupplierContext_best.zip
```

The corresponding score and training step are stored in:

```text
mymodelStrongSupplierContext_best_meta.json
```

Resume training from the newest available checkpoint with:

```bash
RESUME=1 \
RUN_NAME=long_training \
python -m myagent.train 1000000
```

If no checkpoint is found, training starts from the beginning.

## Running on SLURM

Training and benchmark jobs can also be run on a SLURM cluster.

All commands should be submitted from the repository root.

### Standard Training

```bash
sbatch train_scml.sh
```

This trains all six context policies.

### Long Training Run

```bash
sbatch slurm/train_scml_1m.sh
```

The long training script uses one array task per seed and supports checkpointing and resuming.

A different training duration can be passed through environment variables:

```bash
STEPS=3000000 \
sbatch --time=12:00:00 slurm/train_scml_1m.sh
```

Use a unique name for every run:

```bash
sbatch \
  --export=ALL,RUN_NAME=experiment1,STEPS=1000000 \
  slurm/train_scml_1m.sh
```

### Sharded Benchmarks

Submit candidate benchmarks with:

```bash
./slurm/benchmark_candidates.sh
```

This script should be run directly rather than submitted with `sbatch`. It starts the individual benchmark shard jobs.

After all shards have finished, aggregate and re-rank their results:

```bash
python scripts/aggregate_benchmark.py \
  log/benchmark_shards/<RUN_NAME> \
  --run <RUN_NAME>
```

## Configuration

Training and evaluation can be configured through environment variables without changing the source code.

Example:

```bash
TRAIN_CONTEXTS=WeakSupplierContext \
RUN_NAME=experiment1 \
LOG_REWARD_COMPONENTS=1 \
python -m myagent.train 150000
```

### Main Configuration Variables

| Variable              | Default          | Description                                       |
| --------------------- | ---------------- | ------------------------------------------------- |
| `TRAIN_CONTEXTS`      | All six contexts | Contexts to train                                 |
| `RUN_NAME`            | `default`        | Name used for model and log directories           |
| `EVAL_FREQ`           | `steps / 5`      | Timesteps between evaluations                     |
| `N_EVAL_EPISODES`     | `3`              | Number of evaluation worlds                       |
| `CHECKPOINT_FREQ`     | `100000`         | Timesteps between checkpoints                     |
| `MAX_PARALLEL_MODELS` | `3`              | Maximum number of policies trained in parallel    |
| `RESUME`              | `0`              | Resume from the newest checkpoint when set to `1` |
| `MODEL_DIR`           | `myagent/models` | Directory containing the six policies             |
| `ACTION_MANAGER`      | `accept`         | Action representation used during training        |
| `OPPONENT_POOL`       | `default`        | Opponent pool used for training                   |
| `SHARD_SEED`          | Unset            | Seed used for benchmark configuration draws       |

### Opponent Pools

The training opponent pool is selected through `OPPONENT_POOL`.

| Value       | Opponents                                           |
| ----------- | --------------------------------------------------- |
| `default`   | Greedy, RandDist and EqualDist                      |
| `strong`    | Default pool plus Cautious, Suzuka and DistRedist   |
| Custom list | Comma-separated `module.path:ClassName` definitions |

Use the stronger predefined pool with:

```bash
OPPONENT_POOL=strong python -m myagent.train 1000000
```

The training opponent pool and the competition pool used for benchmarking are configured separately.

## Reward Function

The main reward term is based on the change in the agent's normalized score:

```text
balance / initial balance
```

Its default weight is comparatively large because PPO's critic is trained on raw returns.

Optional reward components include:

* unmet trading need,
* shortfall penalties,
* overshoot and disposal costs,
* productivity,
* price quality,
* completed deal volume,
* negotiation engagement,
* realized margin,
* and potential-based reward shaping.

Reward weights are resolved in the following order:

```text
environment variable
        >
per-context default
        >
global default
```

An environment variable therefore overrides the per-context default for every context in the current run.

Potential-based reward shaping supports two potential functions:

| Value       | Description                                         |
| ----------- | --------------------------------------------------- |
| `coverage`  | Fraction of the active trading need already covered |
| `dayprofit` | Realized profit during the current day              |

Reward scales can be inspected with:

```bash
python scripts/measure_reward_scale.py
```

## Logging and Diagnostics

Reward components can be written to per-step CSV files:

```bash
LOG_REWARD_COMPONENTS=1 python -m myagent.train 300000
```

The files are stored in:

```text
log/reward_component_logs/<RUN_NAME>/<context>/<job_id>/
```

Verbose world logging can be enabled for debugging:

```bash
LOG_WORLD=1 python -m myagent.train 300000
```

This stores additional information about contracts and negotiations and allows environment exceptions to propagate.

## Limitations

The agent groups market situations into six predefined contexts. Different situations within the same context are therefore handled by the same specialist policy.

Performance depends on the selected opponent pool and on the distribution of benchmark worlds. Results from different model sets are only directly comparable when they are evaluated with the same tournament settings and seeds.

Training is also sensitive to random seeds, reward scaling and the selected action representation. A model that performs best during an intermediate fixed-world evaluation does not necessarily perform best in a larger tournament.

Full training and benchmarking require substantial CPU time. The included SLURM scripts therefore support parallel training, checkpoints and sharded evaluation.

