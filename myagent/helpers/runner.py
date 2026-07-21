import os
import random
from pathlib import Path
import time

# Hide the GPU before torch is imported. This runner forks parallel workers, and
# any torch-using agent that touches CUDA in a forked child raises "Cannot
# re-initialize CUDA in forked subprocess"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from negmas.helpers import humanize_time
from rich import print
from scml.utils import (
    anac2024_oneshot,
    anac2024_std,
    DefaultAgentsStd2024,
    DefaultAgentsOneShot2024,
)
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt

from scml_agents import get_agents


def load_pool(year, competition="oneshot"):
    """Load a competition year's agent pool for ``competition`` ("oneshot"/"std").

    Prefers the qualified-only set and falls back to the full pool if that
    year's qualification metadata is unusable (mirrors scripts/benchmark.py).
    Note the 2025/2026 pools only exist in the scml-agents *repo*, not the
    released package — see the benchmark notes in the README on installing it
    (--no-deps, to keep the tested scml 0.7.5 stack).
    """
    track = "oneshot" if competition == "oneshot" else "std"
    for qualified in (True, False):
        try:
            agents = list(
                get_agents(
                    year,
                    track=track,
                    qualified_only=qualified,
                    as_class=True,
                    ignore_failing=True,
                )
            )
        except Exception:
            continue
        if agents:
            return agents
    return []


def run(
    competitors=tuple(),
    competition="oneshot",
    reveal_types=True,
    n_steps=20,
    n_configs=2,
    debug=True,
    serial=False,
    year=None,
    n_competitors_per_world=None,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competitors: A list of competitor classes
        competition: The competition type to run (possibilities are oneshot, std).
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try.
                     Different world configurations will correspond to
                     different number of factories, profiles, production graphs etc
        reveal_types: If given, agent names will reveal their type (kind of) and position
        debug: If given, a debug run is used.
        serial: If given, a serial run will be used.
        year:  If given (e.g. 2024/2025/2026), run against that year's competition
               pool (qualified agents) instead of the default winners set. The
               2025/2026 pools require the scml-agents repo installed (see
               load_pool). If omitted, the legacy 2021-2023 winners are used.
        n_competitors_per_world: If given, fix how many competitors share each
               world. Leave unset for small pools; set it (e.g. 2) with a large
               pool like --year 2026, otherwise the round-robin runs every
               C(N, k) combination and the world count explodes.

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value
        - To use breakpoints in your code under pdb, pass both debug=True and serial=True

    """

    if year is not None:
        opponents = load_pool(year, competition)
        if not opponents:
            raise SystemExit(
                f"No {competition} agents found for year {year} — is an "
                f"scml-agents install with that pool available?"
            )
        print(f"Running against the {year} {competition} pool ({len(opponents)} agents).")
    else:
        # Legacy default: the OneShot winners of the last three years.
        opponents = [
            get_agents(y, track="oneshot", winners_only=True, as_class=True)[0]
            for y in (2021, 2022, 2023)
        ]

    # scml_agents runs a module-level random.seed(0) on import, which pins the
    # global RNG that anac2024_*'s config generator draws from — leaving it,
    # every config in this run would be near-identical. Reseed from OS entropy
    # after the pool import and before the tournament (mirrors benchmark.py).
    random.seed()
    np.random.seed()

    if competition == "oneshot":
        competitors = list(competitors) + list(DefaultAgentsOneShot2024) + opponents
    else:
        competitors = list(competitors) + list(DefaultAgentsStd2024) + opponents

    start = time.perf_counter()
    if competition == "std":
        runner = anac2024_std
    else:
        runner = anac2024_oneshot
    results = runner(
        competitors=competitors,
        verbose=True,
        n_steps=n_steps,
        n_configs=n_configs,
        n_competitors_per_world=n_competitors_per_world,
        debug=debug,
        parallelism="serial" if serial else "parallel",
        agent_name_reveals_position=reveal_types,
        agent_name_reveals_type=reveal_types,
    )
    # just make names shorter
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(  # type: ignore
        "."
    ).str[-1]
    # display results
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))  # type: ignore
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")
    #show_score_per_level(results) # type: ignore
        


def show_score_per_level(results):
    results.scores["level"] = results.scores.agent_name.str.split("@", expand=True).loc[
            :, 1
        ]
    results.scores = results.scores.sort_values("level")
    sns.lineplot(
        data=results.scores[["agent_type", "level", "score"]],
        x="level",
        y="score",
        hue="agent_type",
        errorbar=None,
    )
    plt.plot([0.0] * len(results.scores["level"].unique()), "b--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    Path("figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("figures/score_per_level.png", bbox_inches="tight")

if __name__ == "__main__":
    import typer

    typer.run(run)


