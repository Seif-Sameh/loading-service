"""Gymnasium-compatible packing environment.

The environment's sole job is to maintain state and emit action candidates. All business
logic (scoring, constraints) lives in :mod:`app.constraints`, so heuristics and RL share it.

Design notes
------------
- **Action format.** Actions are integers in ``[0, max_candidates)``. The episode's current
  :class:`PackingState` exposes the concrete :class:`CandidateAction` list corresponding to
  those integer indices. The caller is expected to use an action mask; selecting a masked
  action aborts the episode with a negative terminal reward.
- **Observation format.** We expose the raw state object for heuristics and a tensor-dict
  observation for the RL agent. Tensor layout is documented on :class:`PackingState`.
- **Reward.** Incremental — see :mod:`app.constraints.reward`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.constraints.cog import CoGTracker
from app.constraints.mask import build_feasibility_mask
from app.constraints.reward import (
    DEFAULT_REWARD_CFG,
    RewardConfig,
    score_state,
    score_step,
    stability_bearing_delta,
)
from app.env.ems import ExtractConfig, extract_candidate_actions
from app.env.heightmap import Heightmap
from app.schemas import (
    CandidateAction,
    CargoItem,
    Container,
    Placement,
)


@dataclass
class PackingState:
    """Snapshot of the current packing situation handed to every algorithm."""

    container: Container
    heightmap: Heightmap
    placements: list[Placement]
    items_remaining: list[CargoItem]  # in original order; next item = items_remaining[0]
    unplaced: list[CargoItem]  # items the env could not fit (no feasible candidates)
    candidates: list[CandidateAction]  # already feasibility-filtered
    total_weight_kg: float
    cog: CoGTracker
    step_index: int
    seed: int | None

    @property
    def current_item(self) -> CargoItem | None:
        return self.items_remaining[0] if self.items_remaining else None


class PackingEnv(gym.Env):
    """One-container, online 3-D bin-packing environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        container: Container,
        items: list[CargoItem],
        *,
        heightmap_resolution_mm: int = 10,
        max_candidates: int = 80,
        reward_cfg: RewardConfig = DEFAULT_REWARD_CFG,
        seed: int | None = None,
    ) -> None:
        self.container = container
        self.items = items
        self.heightmap_resolution_mm = heightmap_resolution_mm
        self.max_candidates = max_candidates
        self.reward_cfg = reward_cfg
        self.seed = seed

        # Observation / action spaces are used only by the RL agent; heuristics go through
        # the Python state object directly.
        self.action_space = spaces.Discrete(max_candidates)
        self.observation_space = spaces.Dict(
            {
                "ems": spaces.Box(low=0.0, high=1.0, shape=(max_candidates, 6), dtype=np.float32),
                "item": spaces.Box(low=0.0, high=1.0, shape=(2, 3), dtype=np.float32),
                "mask": spaces.MultiBinary(max_candidates),
            }
        )

        # Cached once — used in every step's constraint checks. Avoids rebuilding
        # the dict 4096*32 times per training iteration.
        self._items_by_id = {it.id: it for it in self.items}
        self._state: PackingState | None = None
        self.reset(seed=seed)

    # ----- gym API -----

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        heightmap = Heightmap(self.container, resolution_mm=self.heightmap_resolution_mm)
        self._state = PackingState(
            container=self.container,
            heightmap=heightmap,
            placements=[],
            items_remaining=list(self.items),
            unplaced=[],
            candidates=[],
            total_weight_kg=0.0,
            cog=CoGTracker(container=self.container),
            step_index=0,
            seed=seed,
        )
        self._advance_to_next_fittable()
        return self._obs(), self._info()

    def step(self, action: int):
        assert self._state is not None
        state = self._state

        # Masked action picked: episode ends with a strong negative.
        if action < 0 or action >= len(state.candidates):
            reward = -1.0
            return self._obs(), reward, True, False, self._info()

        chosen = state.candidates[action]
        item = state.items_remaining.pop(0)

        # Apply placement
        placement = Placement(
            item_id=item.id,
            position=chosen.position,
            rotation=chosen.rotation,
            rotated_dimensions=chosen.rotated_dimensions,
        )
        state.placements.append(placement)
        state.heightmap.place(
            placement.position,
            placement.rotated_dimensions.length_mm,
            placement.rotated_dimensions.width_mm,
            placement.rotated_dimensions.height_mm,
        )
        state.total_weight_kg += item.weight_kg
        state.cog.add(placement, item.weight_kg)
        state.step_index += 1

        # O(N) delta — only checks the new placement against items it directly rests on.
        # Replaces two O(N²) scans (current vs previous full state) used in v0.1.
        unstable_added, overloaded_added = stability_bearing_delta(
            placement, state.placements[:-1], self._items_by_id
        )

        terms = score_step(
            placement_volume_mm3=placement.rotated_dimensions.volume_mm3,
            container=self.container,
            cog=state.cog,
            lifo_violation_added=False,  # computed on final state for simplicity
            stack_violation_added=False,
            unstable=unstable_added,
            overloaded=overloaded_added,
            imdg_added=False,  # IMDG can't be added here: mask already excluded infeasibles
        )
        reward = terms.total(self.reward_cfg)

        done = self._advance_to_next_fittable()
        return self._obs(), reward, done, False, self._info()

    # ----- helpers -----

    def _refresh_candidates_for_current(self) -> None:
        """Recompute the candidate list for ``items_remaining[0]`` only."""
        state = self._state
        assert state is not None
        if not state.items_remaining:
            state.candidates = []
            return
        item = state.items_remaining[0]
        cands = extract_candidate_actions(
            item=item,
            item_index=0,
            container=self.container,
            heightmap=state.heightmap,
            placements=state.placements,
            config=ExtractConfig(max_candidates=self.max_candidates),
        )
        mask = build_feasibility_mask(
            candidates=cands,
            item=item,
            container=self.container,
            placed=state.placements,
            items_by_id=self._items_by_id,
            current_total_weight_kg=state.total_weight_kg,
        )
        state.candidates = mask.filter_feasible()

    def _advance_to_next_fittable(self) -> bool:
        """Skip items that cannot be placed in the current state.

        Each unfittable item is moved from ``items_remaining`` to ``unplaced`` so the caller
        sees the same item set in either bucket. Returns True if the episode is now done
        (no more items left at all)."""
        state = self._state
        assert state is not None
        while state.items_remaining:
            self._refresh_candidates_for_current()
            if state.candidates:
                return False
            state.unplaced.append(state.items_remaining.pop(0))
        state.candidates = []
        return True

    def _obs(self) -> dict[str, np.ndarray]:
        """Tensor observation for the RL agent.

        - ``ems``   — (K, 6) of (x, y, z, free_l, free_w, free_h) normalised by container.
        - ``item``  — (2, 3) dimensions for the two upright rotations (next item).
        - ``mask``  — (K,) 1 where the candidate slot is valid.
        """
        state = self._state
        assert state is not None
        K = self.max_candidates
        ems = np.zeros((K, 6), dtype=np.float32)
        mask = np.zeros(K, dtype=np.int8)
        L = self.container.internal.length_mm
        W = self.container.internal.width_mm
        H = self.container.internal.height_mm
        for i, c in enumerate(state.candidates[:K]):
            ems[i, 0] = c.position.x_mm / L
            ems[i, 1] = c.position.y_mm / H
            ems[i, 2] = c.position.z_mm / W
            ems[i, 3] = c.rotated_dimensions.length_mm / L
            ems[i, 4] = c.rotated_dimensions.width_mm / W
            ems[i, 5] = c.rotated_dimensions.height_mm / H
            mask[i] = 1

        if state.items_remaining:
            it = state.items_remaining[0]
            item_tensor = np.array(
                [
                    [
                        it.dimensions.length_mm / L,
                        it.dimensions.width_mm / W,
                        it.dimensions.height_mm / H,
                    ],
                    [
                        it.dimensions.width_mm / L,
                        it.dimensions.length_mm / W,
                        it.dimensions.height_mm / H,
                    ],
                ],
                dtype=np.float32,
            )
        else:
            item_tensor = np.zeros((2, 3), dtype=np.float32)

        return {"ems": ems, "item": item_tensor, "mask": mask}

    def _info(self) -> dict[str, Any]:
        assert self._state is not None
        return {
            "n_placed": len(self._state.placements),
            "n_remaining": len(self._state.items_remaining),
            "weight_used": self._state.total_weight_kg,
        }

    # ----- introspection -----

    @property
    def state(self) -> PackingState:
        assert self._state is not None
        return self._state

    def final_score(self) -> tuple[float, Any]:
        """KPIs + scalar score for the terminal state."""
        state = self._state
        assert state is not None
        kpis, score = score_state(
            container=self.container,
            placements=state.placements,
            items_by_id=self._items_by_id,
        )
        return score, kpis
