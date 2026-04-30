"""Ensemble agent — runs every member algorithm to completion on a fork of the environment,
then **replays** the action sequence of whichever member achieved the best space utilisation.

Why this design rather than a per-step lookahead:

- Per-step rollout scoring needs at least 5-10 step horizons to differentiate good and bad
  placements; for 100-item voyages, that's prohibitively expensive (5-10 × 5 candidates × 100
  steps = thousands of forwards per voyage).
- "Full-voyage replay" is *guaranteed* to be no worse than the best member — the worst case
  is "all members agree, pick any one". The expected case is ~max(member utilisations).
- Compute cost: 6 algorithm runs per voyage instead of 1. On 100-item voyages this is
  ~6× the heuristic cost — about 15 seconds per voyage. Still tractable for the eval suite.

The class follows :class:`PackingAlgorithm` so it slots into ``solve()`` / ``iter_solve()``
without special-casing.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from app.algorithms.base import PackingAlgorithm

if TYPE_CHECKING:
    from app.env.packing_env import PackingEnv, PackingState


class EnsembleAgent(PackingAlgorithm):
    """Replay the action sequence of the best-scoring member algorithm.

    Parameters
    ----------
    ppo_agent
        Optional trained policy (any :class:`PackingAlgorithm`). Adds the model's strengths
        on top of the heuristic ensemble. Pass ``None`` to use heuristics only.
    heuristic_codes
        Which baseline heuristics to include. Defaults to all five.
    """

    code = "ensemble"
    display_name = "Ensemble (best-of-runs)"

    def __init__(
        self,
        ppo_agent: PackingAlgorithm | None = None,
        *,
        heuristic_codes: list[str] | None = None,
    ) -> None:
        from app.algorithms import get_algorithm  # local import — avoid cycle

        self.ppo = ppo_agent
        codes = heuristic_codes or ["bl", "extreme_points", "baf", "bssf", "blsf"]
        # Each member is a (label, factory) — factories so we get fresh instances per voyage.
        self._heuristics: list[tuple[str, PackingAlgorithm]] = [
            (c, get_algorithm(c)) for c in codes
        ]
        self._actions: list[int] = []
        self._idx: int = 0
        self._winner_code: str | None = None

    # ----- public ----------

    def attach_env(self, env) -> None:
        """Called once at the start of solve(). Run every member, keep the best sequence."""
        members: list[tuple[str, PackingAlgorithm]] = [(c, h) for c, h in self._heuristics]
        if self.ppo is not None:
            members.append(("ppo", self.ppo))

        best_util = -1.0
        best_actions: list[int] = []
        best_code = members[0][0]

        for code, algo in members:
            actions = self._simulate(algo, env)
            # Replay actions on a fresh fork to compute final util cleanly
            sim_env = copy.deepcopy(env)
            sim_env.reset()
            sim_env = self._replay(sim_env, actions)
            _score, kpis = sim_env.final_score()
            if kpis.utilization > best_util:
                best_util, best_actions, best_code = kpis.utilization, actions, code

        self._actions = best_actions
        self._idx = 0
        self._winner_code = best_code

    def select(self, state: "PackingState") -> int:
        if self._idx < len(self._actions):
            a = self._actions[self._idx]
            self._idx += 1
            # Defensive bounds clamp — env state may differ slightly under extreme edge cases
            if 0 <= a < len(state.candidates):
                return a
            return 0
        # Past the recorded sequence (shouldn't happen if simulate replayed identically) —
        # fall back to the first feasible candidate.
        return 0 if state.candidates else 0

    # Convenience: which member ended up winning this voyage. Useful for analysis.
    @property
    def winner_code(self) -> str | None:
        return self._winner_code

    # ----- helpers --------

    def _simulate(self, algo: PackingAlgorithm, src_env) -> list[int]:
        """Run ``algo`` on a deepcopy of ``src_env``; return the action sequence."""
        env = copy.deepcopy(src_env)
        # Re-attach env to algo (in case it tracks env internally — e.g. another ensemble)
        algo_local = copy.deepcopy(algo)
        try:
            algo_local.attach_env(env)
        except Exception:
            pass
        # Some algorithms require explicit prep (GA does its own thing, but we don't include
        # it in the default ensemble — it's slow and matches BL anyway).
        actions: list[int] = []
        while True:
            if not env.state.candidates:
                break
            try:
                a = algo_local.select(env.state)
            except Exception:
                break
            actions.append(a)
            _, _, done, _, _ = env.step(a)
            if done:
                break
        return actions

    def _replay(self, env, actions: list[int]):
        """Replay an action sequence on a fresh env. Returns the env after replay."""
        for a in actions:
            if not env.state.candidates:
                break
            if a >= len(env.state.candidates):
                a = 0
            _, _, done, _, _ = env.step(a)
            if done:
                break
        return env
