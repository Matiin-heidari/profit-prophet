#!/bin/bash
# Submit sharded benchmarks for BOTH candidate model sets (flex + accept) in
# one go — no model files are moved; each array reads its set directly via the
# MODEL_DIR env var (see myagent/common.py and candidate_models/README.md).
#
# Run from the repo root on the login node (this is a submitter, not a job):
#   ./slurm/benchmark_candidates.sh              # 5 shards per set
#   SHARDS=10 ./slurm/benchmark_candidates.sh    # 10 shards per set
#   SETS="accept" ./slurm/benchmark_candidates.sh  # only one set
#   MAX_CONCURRENT=3 ./slurm/benchmark_candidates.sh  # <=3 shards running per set
#
# Each set gets its own RUN_NAME (bench_<set>_<timestamp>), so scores land in
# log/benchmark_shards/<RUN_NAME>/ and context-usage logs stay separated.
# After BOTH arrays finish, aggregate each set:
#   python scripts/aggregate_benchmark.py log/benchmark_shards/<RUN_NAME> --run <RUN_NAME>
# and compare the two "MyAgent: rank ... gap_to_best ..." lines.
set -euo pipefail

SHARDS="${SHARDS:-5}"
# Max shards RUNNING at once per set (SLURM array % throttle). Default = all of
# them; total concurrent tasks across a run = MAX_CONCURRENT x number of SETS.

MAX_CONCURRENT="${MAX_CONCURRENT:-$SHARDS}"
SETS="${SETS:-flex accept}"
STAMP="$(date +%m%d_%H%M)"

for set_name in $SETS; do
    model_dir="candidate_models/${set_name}"
    if [ ! -f "${model_dir}/mymodelWeakConsumerContext.zip" ]; then
        echo "ERROR: ${model_dir} is missing models — aborting." >&2
        exit 1
    fi
    run_name="bench_${set_name}_${STAMP}"
    job_id=$(sbatch --parsable \
        --array="0-$((SHARDS - 1))%${MAX_CONCURRENT}" \
        --export=ALL,MODEL_DIR="${model_dir}",RUN_NAME="${run_name}" \
        slurm/benchmark_shard.sh)
    echo "submitted ${set_name}: job ${job_id}  RUN_NAME=${run_name}  MODEL_DIR=${model_dir}"
    echo "  aggregate later:  python scripts/aggregate_benchmark.py log/benchmark_shards/${run_name} --run ${run_name}"
done
