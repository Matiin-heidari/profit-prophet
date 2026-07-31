"""Opponent (non-competitor) pool selection for TRAINING worlds.

The training/eval contexts fill non-agent world slots from
``GeneralContext.non_competitors``, which defaults to the weak
``DefaultAgentsOneShot`` trio (Greedy, RandDist, EqualDist) — a saturated
pool where near-random agents already score ~0.9, while the qualifier
benchmark judges us against much stronger agents
``OPPONENT_POOL=strong`` mixes the top 2024 qualifiers into the pool so the
training distribution matches what the benchmark measures.

This module is deliberately SB3/torch-free (like myagent/action.py) so it
unit-tests without the RL stack; scml_agents is imported lazily and ONLY when
a strong pool is actually requested.

Only make_env (myagent/train.py) consults this. evaluate_model keeps the
unmodified context so the fixed-seed eval curves (0_key/*) stay comparable
with every historical run, and deployment (myagent.py) is untouched.
"""

import importlib
import os
import random

import numpy as np
from scml.oneshot.agents import (
    EqualDistOneShotAgent,
    GreedyOneShotAgent,
    RandDistOneShotAgent,
)

DEFAULT_POOL = (
    GreedyOneShotAgent,
    RandDistOneShotAgent,
    EqualDistOneShotAgent,
)
"""The scml default non-competitor trio (== GeneralContext's own default)."""

STRONG_AGENT_NAMES = (
    "CautiousOneShotAgent",
    "SuzukaAgent",
    "DistRedistAgent",
)
"""Top-3 of the 2024 qualifier pool by pooled mean score in our own sharded
benchmarks (bench_{flex,accept}_0707_1216: Cautious 1.082, Suzuka 1.079,
DistRedist 1.078 over 775 worlds each)."""

STRONG_POOL_YEAR = 2024

_cache: dict[str, tuple | None] = {}


def _reseed_from_os_entropy() -> None:
    """Undo scml_agents' import-time RNG pollution."""
    random.seed()
    np.random.seed()


def _load_strong_agents() -> tuple:
    """Import the curated strong qualifiers from the installed scml_agents."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        from scml_agents import get_agents

        pool = list(
            get_agents(
                STRONG_POOL_YEAR,
                track="oneshot",
                qualified_only=True,
                as_class=True,
                ignore_failing=True,
            )
        )
    finally:
        _reseed_from_os_entropy()

    by_name = {cls.__name__: cls for cls in pool}
    missing = [name for name in STRONG_AGENT_NAMES if name not in by_name]
    if missing:
        raise RuntimeError(
            f"OPPONENT_POOL=strong: {missing} not found in the "
            f"{STRONG_POOL_YEAR} qualifier pool "
            f"(available: {sorted(by_name)})"
        )
    return tuple(by_name[name] for name in STRONG_AGENT_NAMES)


def _import_spec(spec: str) -> type:
    """Import an agent class from a ``module.path:ClassName`` spec."""
    module_name, sep, class_name = spec.partition(":")
    if not sep or not module_name or not class_name:
        raise ValueError(
            f"OPPONENT_POOL entry {spec!r} is not 'module.path:ClassName'"
        )
    try:
        module = importlib.import_module(module_name)
    finally:
        # Arbitrary agent modules may pollute the RNG the same way
        # scml_agents does — always reseed after importing one.
        _reseed_from_os_entropy()
    return getattr(module, class_name)


def resolve_opponent_pool(kind: str | None = None) -> tuple | None:
    """Resolve the non-competitor pool for training worlds.

    ``kind`` (default: the OPPONENT_POOL env var) is one of:

    - ``default`` / empty — return None: leave the context's own default pool
      untouched (byte-identical behavior to before this module existed).
    - ``strong`` — DEFAULT_POOL + the curated top-2024 qualifiers. Mixed, not
      replaced: world generation samples slots from the tuple, and an
      all-strong pool would both over-shift the training distribution and
      slow every world.
    - a comma-separated list of ``module.path:ClassName`` specs — exactly
      those classes (full control for sweeps; NOT auto-mixed with defaults).

    The result is cached per kind — make_env calls this once per env worker,
    and the strong path imports the whole qualifier pool.
    """
    if kind is None:
        kind = os.environ.get("OPPONENT_POOL", "default")
    kind = kind.strip()

    if kind in _cache:
        return _cache[kind]

    if kind in ("", "default"):
        pool = None
    elif kind == "strong":
        pool = DEFAULT_POOL + _load_strong_agents()
    else:
        pool = tuple(
            _import_spec(spec.strip())
            for spec in kind.split(",")
            if spec.strip()
        )
        if not pool:
            pool = None

    _cache[kind] = pool
    return pool


def describe_pool(pool: tuple | None) -> str:
    """Human-readable pool description for config dumps."""
    if pool is None:
        return "context default (Greedy, RandDist, EqualDist)"
    return ", ".join(cls.__name__ for cls in pool)
