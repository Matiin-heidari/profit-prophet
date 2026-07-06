"""Custom RL action managers.

`AcceptFlagActionManager` extends scml's `FlexibleActionManager` with an
explicit per-partner ACCEPT action. Select it for training/eval with the
``ACTION_MANAGER=accept`` environment variable (see `make_action_manager`);
`MyAgent` picks the manager per loaded model automatically by matching the
model's saved action space, so old- and new-space models can coexist.

This module must stay importable without stable_baselines3/torch so the
action-space logic can be unit-tested in a lightweight environment.
"""

from __future__ import annotations

import os

import numpy as np
from gymnasium import spaces
from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse

from scml.oneshot.awi import OneShotAWI
from scml.oneshot.rl.action import ActionManager, FlexibleActionManager
from scml.oneshot.rl.common import group_partners

__all__ = ["AcceptFlagActionManager", "make_action_manager"]


class AcceptFlagActionManager(FlexibleActionManager):
    """`FlexibleActionManager` plus an explicit per-partner ACCEPT value.

    The base manager only emits ACCEPT when the decoded offer happens to
    *exactly equal* the partner's current offer (quantity AND price bin) — a
    needle-in-a-haystack copying task for the policy (CLAUDE.md §10). This
    manager reserves one extra value at the top of each partner's quantity
    dimension:

    - ``q <= max_quantity``    → exactly the base manager's semantics.
    - ``q == max_quantity + 1`` → "close with this partner": ACCEPT their
      current offer if one exists, otherwise END the negotiation (there is
      nothing to accept in a first-proposal round; the flag consistently
      means "I am done negotiating with you").

    The price half of a flagged slot is ignored (the accepted offer is the
    partner's, verbatim). Grouped slots (more real partners than action
    slots) accept every member's current offer. Only discrete action spaces
    are supported.
    """

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.continuous:
            raise NotImplementedError(
                "AcceptFlagActionManager only supports discrete action spaces"
            )

    @property
    def accept_value(self) -> int:
        """The reserved quantity value that signals ACCEPT/close."""
        return self.max_quantity + 1

    def make_space(self) -> spaces.MultiDiscrete:
        """Base space with one extra quantity value per partner for the flag."""
        return spaces.MultiDiscrete(
            np.asarray(
                [self.accept_value + 1, self.n_prices] * self.n_partners
            ).flatten()
        )

    def _partner_groups(self, awi: OneShotAWI) -> list[list[str]]:
        """Partner groups in action-slot order (supplier slots, then consumer
        slots) — must mirror the grouping `recover_offers` uses in decode."""
        return group_partners(
            awi.my_suppliers, self.n_suppliers, self.max_group_size
        ) + group_partners(awi.my_consumers, self.n_consumers, self.max_group_size)

    def decode(self, awi: OneShotAWI, action: np.ndarray) -> dict[str, SAOResponse]:
        action = np.asarray(action)
        action = action.reshape((action.size // 2, 2)).copy()
        flagged = action[:, 0] >= self.accept_value
        # Neutralize flagged quantities so the base decode sees valid values;
        # flagged partners' responses are overridden below.
        action[:, 0] = np.where(flagged, 0, action[:, 0])
        responses = super().decode(awi, action)
        if not flagged.any():
            return responses

        nmis = awi.current_nmis
        for group, is_flagged in zip(self._partner_groups(awi), flagged, strict=True):
            if not is_flagged:
                continue
            for partner in group:
                nmi = nmis.get(partner, None)
                if nmi is None:
                    continue
                offer = nmi.state.current_offer  # same source the base ACCEPT check uses
                if offer is None:
                    responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                else:
                    responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
        return responses

    def encode(self, awi: OneShotAWI, responses: dict[str, SAOResponse]) -> np.ndarray:
        """Inverse of `decode` (used for tests and behavior-cloning labels).

        A slot is flagged if any partner in its group responded ACCEPT_OFFER
        (with grouped partners the counter-offer of a non-accepting group
        member is lost — the unavoidable approximation of grouping).
        """
        encoded = np.asarray(super().encode(awi, responses)).reshape(-1, 2)
        for i, group in enumerate(self._partner_groups(awi)):
            for partner in group:
                response = responses.get(partner, None)
                if (
                    response is not None
                    and response.response == ResponseType.ACCEPT_OFFER
                ):
                    encoded[i, 0] = self.accept_value
                    break
        return encoded.reshape(-1)


def make_action_manager(context, continuous: bool = False) -> ActionManager:
    """Create the action manager selected by the ACTION_MANAGER env var.

    ``flexible`` (default) → scml's `FlexibleActionManager` (the action space
    every model trained before 2026-07-06 uses); ``accept`` →
    `AcceptFlagActionManager`. Training, eval and debug entrypoints must all
    go through this factory so they stay consistent within a run.
    """
    kind = os.environ.get("ACTION_MANAGER", "flexible").strip().lower()
    if kind in ("accept", "accept_flag", "acceptflag"):
        return AcceptFlagActionManager(context=context, continuous=continuous)
    if kind in ("flexible", "default", ""):
        return FlexibleActionManager(context=context, continuous=continuous)
    raise ValueError(f"Unknown ACTION_MANAGER={kind!r} (use 'flexible' or 'accept')")
