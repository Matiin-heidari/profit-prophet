"""Aggregate the PBRS A/B runs: per-arm mean +/- std across seeds, per context.

Reads log/tensorboard_logs/ab_<arm>_s<seed>/<context>/ for arm in {pureprofit, pbrs}
and prints, for each key metric, the two arms' mean +/- std at each eval step,
plus a paired (per-seed) comparison so we can tell a real effect from noise.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from myagent.common import LOG_ROOT

ARMS = ["pureprofit", "pbrs"]
SEEDS = [0, 1, 2]
CONTEXTS = ["StrongSupplierContext", "WeakSupplierContext"]
METRICS = [
    "0_key/score",
    "0_key/score_gap_vs_best_opponent",
    "0_key/my_rank",
]


def load(run, ctx, tag):
    ds = glob.glob(f"{LOG_ROOT}/tensorboard_logs/{run}/{ctx}/*/")
    if not ds:
        return None
    ea = EventAccumulator(ds[0], size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def arm_matrix(arm, ctx, tag):
    """Stack the metric across seeds -> (n_seeds, n_points); None if missing."""
    rows, steps = [], None
    for s in SEEDS:
        got = load(f"ab_{arm}_s{s}", ctx, tag)
        if got is None:
            return None, None
        st, v = got
        if steps is None:
            steps = st
        rows.append(v[: len(steps)])
    m = min(len(r) for r in rows)
    return steps[:m], np.stack([r[:m] for r in rows])


def main():
    for ctx in CONTEXTS:
        print(f"\n################ {ctx} ################")
        for tag in METRICS:
            steps_pp, pp = arm_matrix("pureprofit", ctx, tag)
            steps_pb, pb = arm_matrix("pbrs", ctx, tag)
            if pp is None or pb is None:
                print(f"  {tag}: missing data"); continue
            n = min(pp.shape[1], pb.shape[1])
            pp, pb, steps = pp[:, :n], pb[:, :n], steps_pp[:n]
            better = "higher" if tag == "0_key/score" or "gap" in tag else "lower"
            print(f"\n  == {tag}  (better = {better}) ==")
            print(f"    {'step':>8} {'pureprofit(mean±std)':>24} {'pbrs(mean±std)':>24}  {'Δ(pbrs-pp)':>10}")
            for i in (0, n // 2, n - 1):  # first / mid / last
                a, b = pp[:, i], pb[:, i]
                print(f"    {int(steps[i]):>8} "
                      f"{a.mean():>+10.4f}±{a.std():.4f}      "
                      f"{b.mean():>+10.4f}±{b.std():.4f}   {b.mean()-a.mean():>+8.4f}")
            # paired (per-seed) final-point comparison
            fa, fb = pp[:, -1], pb[:, -1]
            dpaired = fb - fa
            sign = np.sign(dpaired)
            print(f"    paired final Δ per seed (pbrs-pp): "
                  f"{np.round(dpaired,4).tolist()}   mean={dpaired.mean():+.4f}")
            # area-under-curve (sample efficiency: average level over training)
            auc_a, auc_b = pp.mean(axis=1), pb.mean(axis=1)
            print(f"    mean-over-training (AUC) per seed: pp={np.round(auc_a,4).tolist()}  "
                  f"pbrs={np.round(auc_b,4).tolist()}  Δmean={auc_b.mean()-auc_a.mean():+.4f}")


if __name__ == "__main__":
    main()
