"""Plot metrics from the logBackups/csv_exports/ CSVs, with cross-experiment overlay.

Reads the per-context CSVs produced for the experiment archive
(logBackups/csv_exports/<experiment>/<Context>.csv, columns `step` +
`s<seed>_<metric>` for multi-seed runs or bare `<metric>` for single-run
experiments) and plots **one figure per --variable**. Every --experiment and
every --context you pass is overlaid on that same figure -- pass several of
either (or both) to combine/compare them on one graph.

Each experiment has its own base color, drawn from a curated, pairwise
checked palette (not just evenly-spaced hues, which can put sort-adjacent
experiments too close together -- see _CURATED_PALETTE_HEX); each context is
a fixed, small offset within that family (see build_color_registry /
color_for). So: an experiment's lines always look like the same family
across plots showing different contexts; a given context always sits at the
same relative spot within any experiment's family; and every
(experiment, context) pair still gets a color that's fixed, consistent
across every plot and every invocation of this script, and distinct from
every other pair's. All seeds in a group share that exact color and one
legend entry.

Shortcuts
---------
--experiment accepts, besides the full folder name:
  * a leading number, e.g. `-e 02` for `02_pureprofit_baseline`. If the number
    covers several sub-experiments (e.g. `04` has four reward-shaping arms),
    all of them are included.
  * a case-insensitive substring, e.g. `-e premargin` or `-e pbrs_multi`.
--context accepts, besides the full name (e.g. StrongSupplierContext):
  * a two-letter code: SS/SC/BS/BC/WS/WC (strength+side initials).
  * a compact/underscored name: strongsupplier, weak_consumer, etc.
  * "all" (or "*") for every context -> one plot per context.
Run with --list to print every available experiment (with its shortcut
number) and context (with its two-letter code).

Examples
--------
# One experiment, one context, one variable (3 seed lines, one plot)
python -m scripts.plot_experiments -e 02_pureprofit_baseline -c StrongSupplierContext -v score

# Same thing, using shortcuts: "02" for the experiment, "SS" for the context
python -m scripts.plot_experiments -e 02 -c SS -v score

# Compare pure-profit baseline vs pre-margin reward vs new-PPO, same context/variable
# -> one plot, 9 lines (3 experiments x 3 seeds)
python -m scripts.plot_experiments -e 02 -e 03 -e 06 -c SS -v score

# Combine two contexts onto the same plot -> one plot, 6 lines (2 contexts x 3 seeds)
python -m scripts.plot_experiments -e 02 -c SS -c WC -v score

# All six contexts combined into one plot; -v repeated -> one plot per variable
python -m scripts.plot_experiments -e 02 -e 06 -c all -v score -v my_rank

# See all available experiment/context shortcuts
python -m scripts.plot_experiments --list
"""

import argparse
import colorsys
import hashlib
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV_ROOT_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logBackups", "csv_exports",
)
OUT_DIR_DEFAULT = os.path.join(CSV_ROOT_DEFAULT, "plots")

SEED_COL_RE = re.compile(r"^s(\d+)_(.+)$")

ALL_CONTEXTS = [
    "StrongSupplierContext", "StrongConsumerContext",
    "BalancedSupplierContext", "BalancedConsumerContext",
    "WeakSupplierContext", "WeakConsumerContext",
]


def _split_context_name(name: str):
    base = name[: -len("Context")] if name.endswith("Context") else name
    for side in ("Supplier", "Consumer"):
        if base.endswith(side):
            return base[: -len(side)], side
    return base, ""


def _build_context_aliases() -> dict:
    """Map lowercase shortcuts (two-letter code, compact name, snake_case) -> full context name."""
    aliases = {}
    for name in ALL_CONTEXTS:
        strength, side = _split_context_name(name)
        code = (strength[:1] + side[:1]).lower()          # "ss", "bc", "wc", ...
        aliases[code] = name
        aliases[f"{strength}{side}".lower()] = name        # "strongsupplier"
        aliases[f"{strength}_{side}".lower()] = name        # "strong_supplier"
    return aliases


CONTEXT_ALIASES = _build_context_aliases()


def resolve_context(token: str) -> list:
    """Resolve a context name/shortcut to a list of full context names.

    "all" (or "*") expands to every context. Otherwise resolves to a single
    full name (exact match, two-letter code SS/SC/BS/BC/WS/WC, or a compact/
    snake_case name like weaksupplier / weak_consumer). Returns [] if nothing
    matches.
    """
    if token.lower() in ("all", "*"):
        return list(ALL_CONTEXTS)
    if token in ALL_CONTEXTS:
        return [token]
    resolved = CONTEXT_ALIASES.get(token.lower())
    return [resolved] if resolved else []


def discover_leaf_experiments(csv_root: str) -> list:
    """Find every experiment folder that directly holds context CSVs (excludes plots/)."""
    leaves = []
    for dirpath, dirnames, filenames in os.walk(csv_root):
        if dirpath == csv_root:
            dirnames[:] = [d for d in dirnames if d != "plots"]
        if any(f.endswith(".csv") for f in filenames):
            rel = os.path.relpath(dirpath, csv_root).replace(os.sep, "/")
            leaves.append(rel)
    return sorted(leaves)


def _experiment_number(leaf: str):
    m = re.match(r"^(\d+)", leaf.split("/")[0])
    return int(m.group(1)) if m else None


def resolve_experiment(token: str, leaves: list) -> list:
    """Resolve an experiment name/shortcut to one or more leaf experiment paths.

    Accepts: an exact leaf path; a leading number (e.g. "02", matches every
    leaf under that numbered top-level folder -- there can be several, as
    with 04's four reward-shaping arms); or a case-insensitive substring of
    the leaf path. Returns [] if nothing matches.
    """
    if token in leaves:
        return [token]
    if token.isdigit():
        n = int(token)
        matches = [l for l in leaves if _experiment_number(l) == n]
        if matches:
            return matches
    tok_low = token.lower()
    return [l for l in leaves if tok_low in l.lower()]


_CONTEXT_HUE_BAND = 0.03    # total hue spread (turns) across a family's 6 contexts
_CONTEXT_VALUE_STEP = 0.05  # brightness step per context, from the family's base value

# Evenly-spacing hues by sorted-index order (the previous scheme) puts
# alphabetically-adjacent experiments on hue-adjacent colors -- fine most of
# the time, but occasionally two neighbors (e.g. a purple next to a magenta)
# read as near-identical. Sidestep that by assigning experiments from a
# curated, hand-picked palette instead of a formula: every entry here was
# checked pairwise in RGB space (min pairwise distance ~95/441, min
# *consecutive*-entry distance ~130/441 -- consecutive matters most, since
# that's what sort-adjacent experiments actually get) rather than merely
# "looks fine" by eye. Assigned in this fixed order by sorted experiment
# index, so it's a pure function of the experiment list -- no cache file,
# identical across invocations. If more experiments exist than colors here,
# the overflow falls back to an evenly-spaced hue (rare in practice; extend
# the palette if that starts happening).
_CURATED_PALETTE_HEX = [
    "#e6194b",  # red
    "#4363d8",  # blue
    "#8fce00",  # lime
    "#f032e6",  # magenta
    "#008080",  # teal
    "#f58231",  # orange
    "#000075",  # navy
    "#3cb44b",  # green
    "#9a6324",  # brown
    "#17becf",  # cyan
    "#911eb4",  # purple
    "#800000",  # maroon
]


def _hex_to_hsv(hexcode: str) -> tuple:
    r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hsv(r, g, b)


_CURATED_PALETTE_HSV = [_hex_to_hsv(h) for h in _CURATED_PALETTE_HEX]


def build_color_registry(leaves: list) -> dict:
    """Assign every experiment its own base color from the curated palette
    (falling back to an evenly-spaced hue past the palette's length).

    Context is layered on top of an experiment's base color as a small,
    fixed hue+brightness offset (see color_for) rather than getting an
    unrelated color of its own. That gives three properties at once:
      - an experiment's colors always form one recognizable "family",
        whichever contexts happen to be plotted alongside it, so the same
        experiment reads as the same family across plots with different
        context selections;
      - within that family, a given context always sits at the same
        relative position (e.g. StrongSupplierContext is always the family's
        lowest-hue member), so context is *also* consistent across plots;
      - every (experiment, context) pair is still guaranteed distinct: base
        colors are pairwise well-separated (see _CURATED_PALETTE_HEX), and
        the 6 contexts within one family each get a distinct small offset.
    This is a pure function of the sorted experiment list, so it needs no
    cache file and reproduces identically across separate invocations.
    """
    leaves = sorted(leaves)
    n = len(leaves)
    registry = {}
    for i, leaf in enumerate(leaves):
        if i < len(_CURATED_PALETTE_HSV):
            registry[leaf] = _CURATED_PALETTE_HSV[i]
        else:
            registry[leaf] = (i / n if n else 0.0, 0.65, 0.80)
    return registry


def color_for(registry: dict, experiment: str, context: str) -> tuple:
    """Look up an (experiment, context) line's color: the experiment's base
    color (from `registry`), nudged by a small fixed hue+brightness offset
    for `context`'s position among ALL_CONTEXTS, so the 6 contexts stay
    visually distinct within one family without leaving it.

    Every seed in the same experiment+context is drawn in this exact color --
    they're the same run family, just different seeds -- so there's one
    color (and one legend entry) per group rather than per line.
    """
    base = registry.get(experiment)
    if base is None:
        # Shouldn't happen (registry is built from the same experiment list
        # used everywhere else), but don't crash a plot over it.
        key = f"{experiment}\x1f{context}"
        digest = hashlib.md5(key.encode()).digest()
        base = (int.from_bytes(digest[:4], "big") / 2**32, 0.65, 0.80)
    base_hue, sat, base_value = base
    idx = ALL_CONTEXTS.index(context) if context in ALL_CONTEXTS else 2.5
    centered = idx - (len(ALL_CONTEXTS) - 1) / 2  # e.g. -2.5 .. +2.5 over 6 contexts
    hue = (base_hue + centered * _CONTEXT_HUE_BAND / len(ALL_CONTEXTS)) % 1.0
    value = min(0.97, max(0.25, base_value + centered * _CONTEXT_VALUE_STEP))
    return colorsys.hsv_to_rgb(hue, sat, value)


def load_context_csv(csv_root: str, experiment: str, context: str) -> pd.DataFrame | None:
    path = os.path.join(csv_root, experiment, f"{context}.csv")
    if not os.path.isfile(path):
        print(f"  [skip] no such CSV: {path}", file=sys.stderr)
        return None
    return pd.read_csv(path, index_col="step")


def seed_columns(df: pd.DataFrame, variable: str):
    """Yield (seed_label, column_name) pairs for `variable` in `df`.

    Multi-seed CSVs have columns like s0_score/s1_score/s2_score. Single-run
    CSVs (e.g. the per-context or single-seed 400k arms) have a bare `score`
    column instead -> seed_label "single".
    """
    found = []
    for col in df.columns:
        m = SEED_COL_RE.match(col)
        if m and m.group(2) == variable:
            found.append((f"s{m.group(1)}", col))
    if found:
        found.sort(key=lambda t: int(t[0][1:]))
        return found
    if variable in df.columns:
        return [("single", variable)]
    return []


def slugify(experiment: str) -> str:
    return experiment.replace("/", "-")


def context_code(name: str) -> str:
    strength, side = _split_context_name(name)
    return (strength[:1] + side[:1]).upper()


def _short_list(items, limit=3) -> str:
    return ", ".join(items) if len(items) <= limit else f"{len(items)} selected"


def plot_one(csv_root, out_dir, registry, experiments, contexts, variable, seeds_filter, dpi):
    """One combined plot: every (experiment, context, seed) line requested is
    overlaid on the same axes. Color comes from `registry` (see
    build_color_registry) keyed by the (experiment, context) group -- all its
    seeds share the exact same color -- and the legend gets exactly one entry
    per group (not per seed line), labeling the group's first line and
    suppressing the rest via matplotlib's "_nolegend_" prefix.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_line = False
    multi_exp = len(experiments) > 1
    multi_ctx = len(contexts) > 1

    for experiment in experiments:
        for context in contexts:
            df = load_context_csv(csv_root, experiment, context)
            if df is None:
                continue
            cols = seed_columns(df, variable)
            if not cols:
                print(
                    f"  [skip] {experiment}/{context}.csv has no '{variable}' column",
                    file=sys.stderr,
                )
                continue
            color = color_for(registry, experiment, context)
            parts = []
            if multi_exp:
                parts.append(experiment)
            if multi_ctx:
                parts.append(context)
            group_label = " · ".join(parts) if parts else experiment
            labeled = False
            for seed_label, col in cols:
                if seeds_filter and seed_label not in seeds_filter:
                    continue
                series = df[col].dropna()
                if series.empty:
                    print(
                        f"  [skip] {experiment}/{context} {col}: all-NaN (metric never logged)",
                        file=sys.stderr,
                    )
                    continue
                label = group_label if not labeled else "_nolegend_"
                labeled = True
                ax.plot(
                    series.index, series.values,
                    color=color, marker="o", markersize=3, linewidth=1.6, label=label,
                )
                any_line = True

    if not any_line:
        plt.close(fig)
        print(f"  [skip] {variable}: nothing to plot")
        return None

    ax.set_xlabel("training step")
    ax.set_ylabel(variable)
    ax.set_title(f"{variable}\n{_short_list(contexts)}  |  {_short_list(experiments)}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Legend goes *outside* the axes (to the right) so it never covers data,
    # no matter how many lines are on the plot. bbox_inches="tight" at save
    # time expands the saved image to fit it rather than clipping it.
    n_lines = len(ax.get_lines())
    ncol = (n_lines - 1) // 20 + 1  # wrap into more columns instead of growing very tall
    ax.legend(
        fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0, ncol=ncol,
    )

    os.makedirs(out_dir, exist_ok=True)
    exp_slug = "+".join(slugify(e) for e in experiments)
    ctx_slug = "+".join(context_code(c) for c in contexts)
    out_path = os.path.join(out_dir, f"{variable}__e-{exp_slug}__c-{ctx_slug}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "-e", "--experiment", action="append", dest="experiments",
        help="Experiment folder under csv-root (e.g. 02_pureprofit_baseline or "
             "04_reward_shaping_evaluation/pbrs_multiseed), OR a shortcut: a "
             "leading number (02) or a substring (premargin). Repeat to combine "
             "multiple experiments onto the same plot.",
    )
    ap.add_argument(
        "-c", "--context", action="append", dest="contexts",
        help="Context name (e.g. StrongSupplierContext), OR a shortcut: a "
             "two-letter code (SS/SC/BS/BC/WS/WC), compact name (weaksupplier), "
             "or 'all' for every context. Repeat to combine multiple contexts "
             "onto the same plot.",
    )
    ap.add_argument(
        "-v", "--variable", action="append", dest="variables",
        help="Metric column to plot, e.g. score, my_rank, my_score, "
             "best_opponent_score, opponent_score_mean, score_gap, "
             "score_gap_vs_best_opponent, ep_rew_mean. Repeat for one plot per "
             "variable (each combining all requested experiments/contexts).",
    )
    ap.add_argument(
        "--seed", action="append", dest="seeds",
        help="Restrict to these seed labels (e.g. s0 s1). Default: all seeds present.",
    )
    ap.add_argument("--csv-root", default=CSV_ROOT_DEFAULT, help="Root of the per-context CSVs.")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT, help="Where to write PNGs.")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--list", action="store_true",
        help="Print every available experiment (with its shortcut number) and "
             "context (with its two-letter code), then exit.",
    )
    args = ap.parse_args()

    leaves = discover_leaf_experiments(args.csv_root)

    if args.list:
        print("Experiments (under", args.csv_root + "):")
        for l in leaves:
            n = _experiment_number(l)
            print(f"  {n if n is not None else '--':>2}  {l}")
        print("\nContexts:")
        for name in ALL_CONTEXTS:
            strength, side = _split_context_name(name)
            code = (strength[:1] + side[:1]).upper()
            print(f"  {code}  {name}")
        return

    if not args.experiments or not args.contexts or not args.variables:
        ap.error("-e/--experiment, -c/--context and -v/--variable are all required (or pass --list)")

    # Resolve experiment shortcuts -> leaf paths, preserving order, de-duplicated.
    experiments = []
    for token in args.experiments:
        matches = resolve_experiment(token, leaves)
        if not matches:
            ap.error(
                f"no experiment matches '{token}'. Available: "
                + ", ".join(leaves) + "  (see --list)"
            )
        if len(matches) > 1:
            print(f"[resolve] experiment '{token}' -> {', '.join(matches)}")
        for m in matches:
            if m not in experiments:
                experiments.append(m)

    # Resolve context shortcuts -> full names, preserving order, de-duplicated.
    contexts = []
    for token in args.contexts:
        matches = resolve_context(token)
        if not matches:
            ap.error(
                f"no context matches '{token}'. Available: "
                + ", ".join(ALL_CONTEXTS) + ", all  (see --list)"
            )
        if matches != [token]:
            print(f"[resolve] context '{token}' -> {', '.join(matches)}")
        for m in matches:
            if m not in contexts:
                contexts.append(m)

    seeds_filter = set(args.seeds) if args.seeds else None
    registry = build_color_registry(leaves)

    written = []
    for variable in args.variables:
        print(f"[plot] {variable}  <- experiments: {', '.join(experiments)}; contexts: {', '.join(contexts)}")
        out = plot_one(
            args.csv_root, args.out_dir, registry, experiments, contexts, variable,
            seeds_filter, args.dpi,
        )
        if out:
            written.append(out)
            print(f"  -> {out}")

    print(f"\n{len(written)} plot(s) written to {args.out_dir}")


if __name__ == "__main__":
    main()
