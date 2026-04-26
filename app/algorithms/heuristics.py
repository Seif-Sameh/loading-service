"""Constraint-aware, 3-D-aware geometric heuristics.

Every candidate action has already been filtered through :mod:`app.constraints.mask`, so
these scorers only need to pick *among* the feasible options.

All heuristics are deterministic; ties broken by (y, x, z) to make results reproducible.
"""
from __future__ import annotations

from app.algorithms.base import PackingAlgorithm
from app.env.packing_env import PackingState
from app.schemas import CandidateAction


def _free_side_metrics(state: PackingState, c: CandidateAction) -> tuple[int, int, int]:
    """Return (remaining length, remaining width, remaining volume) *below* this candidate.

    The "remaining" room is measured against the container's internal dimensions minus
    the candidate's footprint at its resting (x, z). Imperfect — a real EMS extractor would
    give exact free rectangles — but cheap and monotonic in the same direction, so the
    ordering these heuristics produce matches DeepPack3D's in spirit.
    """
    cont = state.container.internal
    dx = cont.length_mm - (c.position.x_mm + c.rotated_dimensions.length_mm)
    dz = cont.width_mm - (c.position.z_mm + c.rotated_dimensions.width_mm)
    dy = cont.height_mm - (c.position.y_mm + c.rotated_dimensions.height_mm)
    return max(0, dx), max(0, dz), max(0, dy)


def _tiebreak(c: CandidateAction) -> tuple[int, int, int]:
    return c.position.y_mm, c.position.x_mm, c.position.z_mm


class BottomLeft(PackingAlgorithm):
    """Pick the placement with the smallest y, then smallest x, then smallest z.

    This is the classic Bottom-Left heuristic generalised to 3D (z added for tie-break).
    Strong against shallow bins; weak at weight distribution.
    """

    code = "bl"
    display_name = "Bottom-Left"

    def select(self, state: PackingState) -> int:
        return min(range(len(state.candidates)), key=lambda i: _tiebreak(state.candidates[i]))


class BestAreaFit(PackingAlgorithm):
    """Prefer placements that leave the *smallest* free area on top of the item.

    "Area" here is remaining length × remaining width at the item's resting level — the
    3D generalisation of DeepPack3D's 2D `best_area_fit`.
    """

    code = "baf"
    display_name = "Best Area Fit"

    def select(self, state: PackingState) -> int:
        def score(c: CandidateAction) -> tuple[int, tuple[int, int, int]]:
            dx, dz, _ = _free_side_metrics(state, c)
            return dx * dz, _tiebreak(c)

        return min(range(len(state.candidates)), key=lambda i: score(state.candidates[i]))


class BestShortestSideFit(PackingAlgorithm):
    """Minimise the shorter of the two horizontal leftover sides.

    Tends to create long strips of usable space.
    """

    code = "bssf"
    display_name = "Best Shortest Side Fit"

    def select(self, state: PackingState) -> int:
        def score(c: CandidateAction) -> tuple[int, tuple[int, int, int]]:
            dx, dz, _ = _free_side_metrics(state, c)
            return min(dx, dz), _tiebreak(c)

        return min(range(len(state.candidates)), key=lambda i: score(state.candidates[i]))


class BestLongestSideFit(PackingAlgorithm):
    """Minimise the longer of the two horizontal leftover sides."""

    code = "blsf"
    display_name = "Best Longest Side Fit"

    def select(self, state: PackingState) -> int:
        def score(c: CandidateAction) -> tuple[int, tuple[int, int, int]]:
            dx, dz, _ = _free_side_metrics(state, c)
            return max(dx, dz), _tiebreak(c)

        return min(range(len(state.candidates)), key=lambda i: score(state.candidates[i]))


class ExtremePoints(PackingAlgorithm):
    """Extreme-Points heuristic (Crainic, Perboli, Tadei, INFORMS 2008).

    Among feasible candidates, pick the "most extreme" corner — the one closest to (0, 0, 0)
    in a lexicographic sense weighted by container scale. Balances compactness in all three
    axes. Treated in CLP literature as the de-facto modern heuristic baseline.
    """

    code = "extreme_points"
    display_name = "Extreme Points"

    def select(self, state: PackingState) -> int:
        cont = state.container.internal

        def score(c: CandidateAction) -> float:
            # Combined normalised distance from the origin corner.
            return (
                c.position.y_mm / max(cont.height_mm, 1)
                + c.position.x_mm / max(cont.length_mm, 1) * 0.5
                + c.position.z_mm / max(cont.width_mm, 1) * 0.25
            )

        return min(range(len(state.candidates)), key=lambda i: score(state.candidates[i]))
