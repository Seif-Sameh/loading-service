"""Soft scoring — reward terms fed to the RL agent and used to rank heuristic / GA outputs.

The top-level entry points are:

- :func:`score_step` — delta reward for a single placement (used during RL training).
- :func:`score_state` — scalar score for a finished or in-progress solve (used for GA
  fitness and for the "which algorithm won" comparison in the UI).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.constraints.cog import CoGTracker
from app.constraints.imdg import imdg_violations
from app.schemas import CargoItem, Container, KPIs, Placement


# ---------------------------------------------------------------------------
# Thresholds (hansatic)
# ---------------------------------------------------------------------------
# Soft penalty starts beyond these warning bands; gets quadratically worse.
LONG_WARN = 0.10   # ±10 % longitudinal deviation
LAT_WARN = 0.05    # ±5 % lateral deviation
VERT_WARN = 0.40   # vertical CoG fraction above this is penalised


@dataclass(frozen=True)
class RewardConfig:
    """Tunable weights for every soft term."""

    w_util: float = 1.0
    w_cog_long: float = 0.3
    w_cog_lat: float = 0.3
    w_cog_vert: float = 0.3
    w_stability: float = 0.2
    w_bearing: float = 0.2
    w_lifo: float = 0.4
    w_stack: float = 0.2
    w_imdg: float = 1.0


DEFAULT_REWARD_CFG = RewardConfig()


@dataclass
class RewardTerms:
    """Decomposed reward for one step or one episode (handy for debugging / thesis plots)."""

    utilization_delta: float = 0.0
    cog_long_penalty: float = 0.0
    cog_lat_penalty: float = 0.0
    cog_vert_penalty: float = 0.0
    stability_penalty: float = 0.0
    bearing_penalty: float = 0.0
    lifo_penalty: float = 0.0
    stack_penalty: float = 0.0
    imdg_penalty: float = 0.0

    def total(self, cfg: RewardConfig = DEFAULT_REWARD_CFG) -> float:
        return (
            cfg.w_util * self.utilization_delta
            - cfg.w_cog_long * self.cog_long_penalty
            - cfg.w_cog_lat * self.cog_lat_penalty
            - cfg.w_cog_vert * self.cog_vert_penalty
            - cfg.w_stability * self.stability_penalty
            - cfg.w_bearing * self.bearing_penalty
            - cfg.w_lifo * self.lifo_penalty
            - cfg.w_stack * self.stack_penalty
            - cfg.w_imdg * self.imdg_penalty
        )


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _quadratic_beyond(value: float, warn: float) -> float:
    """Zero if |value| <= warn, else quadratic in the excess."""
    excess = max(0.0, abs(value) - warn)
    return excess * excess


def _quadratic_above(value: float, threshold: float) -> float:
    """Zero if value <= threshold, else quadratic in the (signed) excess.

    Used for one-sided penalties — e.g. vertical CoG only matters when *too high*."""
    excess = max(0.0, value - threshold)
    return excess * excess


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_step(
    *,
    placement_volume_mm3: int,
    container: Container,
    cog: CoGTracker,
    lifo_violation_added: bool,
    stack_violation_added: bool,
    unstable: bool,
    overloaded: bool,
    imdg_added: bool,
) -> RewardTerms:
    """Delta reward terms from placing one item.

    The caller owns the CoG tracker (which is updated **before** this is called) and is
    responsible for computing the stability / IMDG / load-bearing flags via the same
    helpers used in :func:`score_state`.
    """
    util_delta = placement_volume_mm3 / container.internal.volume_mm3
    return RewardTerms(
        utilization_delta=util_delta,
        cog_long_penalty=_quadratic_beyond(cog.longitudinal_deviation, LONG_WARN),
        cog_lat_penalty=_quadratic_beyond(cog.lateral_deviation, LAT_WARN),
        cog_vert_penalty=_quadratic_above(cog.vertical_fraction, VERT_WARN),
        stability_penalty=1.0 if unstable else 0.0,
        bearing_penalty=1.0 if overloaded else 0.0,
        lifo_penalty=1.0 if lifo_violation_added else 0.0,
        stack_penalty=1.0 if stack_violation_added else 0.0,
        imdg_penalty=1.0 if imdg_added else 0.0,
    )


def score_state(
    *,
    container: Container,
    placements: list[Placement],
    items_by_id: dict[str, CargoItem],
) -> tuple[KPIs, float]:
    """Return (KPIs, scalar score) for the current state. Used for fitness / comparison."""
    cog = CoGTracker(container=container)
    total_weight = 0.0
    total_volume = 0
    unstable_count = 0
    overloaded_count = 0
    stack_violations = 0
    lifo_violations = 0

    for p in placements:
        it = items_by_id[p.item_id]
        cog.add(p, it.weight_kg)
        total_weight += it.weight_kg
        total_volume += p.rotated_dimensions.volume_mm3

    # pairwise checks on final state
    imdg_count = imdg_violations(placements, items_by_id)
    lifo_violations = _count_lifo_violations(placements, items_by_id, container)
    stack_violations = _count_stack_violations(placements, items_by_id)
    unstable_count, overloaded_count = _count_stability_bearing(placements, items_by_id)

    kpis = KPIs(
        utilization=total_volume / container.internal.volume_mm3,
        weight_used=total_weight / container.payload_kg,
        cog_long_dev=cog.longitudinal_deviation,
        cog_lat_dev=cog.lateral_deviation,
        cog_vert_frac=cog.vertical_fraction,
        unstable_count=unstable_count,
        overloaded_count=overloaded_count,
        imdg_violation_count=imdg_count,
        lifo_violation_count=lifo_violations,
        stack_violation_count=stack_violations,
    )

    cfg = DEFAULT_REWARD_CFG
    score = (
        cfg.w_util * kpis.utilization
        - cfg.w_cog_long * _quadratic_beyond(kpis.cog_long_dev, LONG_WARN)
        - cfg.w_cog_lat * _quadratic_beyond(kpis.cog_lat_dev, LAT_WARN)
        - cfg.w_cog_vert * _quadratic_above(kpis.cog_vert_frac, VERT_WARN)
        - cfg.w_stability * unstable_count
        - cfg.w_bearing * overloaded_count
        - cfg.w_lifo * lifo_violations
        - cfg.w_stack * stack_violations
        - cfg.w_imdg * imdg_count
    )
    return kpis, score


# ---------------------------------------------------------------------------
# Helpers shared between score_state and score_step (used by the environment too)
# ---------------------------------------------------------------------------


def _count_lifo_violations(
    placements: list[Placement], items_by_id: dict[str, CargoItem], container: Container
) -> int:
    """LIFO rule: items for earlier drops must be closer to the door (= smaller x).

    This container convention: door at x = 0, far end at x = internal.length_mm. The "first"
    drop (delivery_stop = 1) should be loaded last (smallest x). An item blocks another if
    its x-range overlaps and its delivery_stop is larger.
    """
    violations = 0
    for i in range(len(placements)):
        pi = placements[i]
        stop_i = items_by_id[pi.item_id].delivery_stop
        if stop_i == 0:
            continue
        for j in range(len(placements)):
            if i == j:
                continue
            pj = placements[j]
            stop_j = items_by_id[pj.item_id].delivery_stop
            if stop_j == 0 or stop_j <= stop_i:
                continue
            # item j is for a later stop. If j blocks i from door access, count a violation.
            z_overlap = not (pj.z_max_mm <= pi.position.z_mm or pi.z_max_mm <= pj.position.z_mm)
            y_overlap = not (pj.y_max_mm <= pi.position.y_mm or pi.y_max_mm <= pj.position.y_mm)
            if z_overlap and y_overlap and pj.position.x_mm < pi.position.x_mm:
                violations += 1
                break
    return violations


def _count_stack_violations(
    placements: list[Placement], items_by_id: dict[str, CargoItem]
) -> int:
    """A stack violation occurs if a heavier item sits above a lighter one."""
    violations = 0
    for i, top in enumerate(placements):
        top_item = items_by_id[top.item_id]
        for j, bot in enumerate(placements):
            if i == j:
                continue
            if bot.y_max_mm != top.position.y_mm:
                continue
            # overlap in x, z?
            ox = max(0, min(top.x_max_mm, bot.x_max_mm) - max(top.position.x_mm, bot.position.x_mm))
            oz = max(0, min(top.z_max_mm, bot.z_max_mm) - max(top.position.z_mm, bot.position.z_mm))
            if ox > 0 and oz > 0:
                bot_item = items_by_id[bot.item_id]
                if top_item.weight_kg > bot_item.weight_kg * 1.05:
                    violations += 1
                    break
    return violations


def _count_stability_bearing(
    placements: list[Placement], items_by_id: dict[str, CargoItem]
) -> tuple[int, int]:
    """Rough support-area + load-bearing check.

    - Stability: item is unstable if < 70 % of its base area is supported by items directly
      below or by the container floor (y == 0).
    - Load-bearing: supporting item is overloaded if top item's pressure exceeds its
      ``crush_strength_kpa`` (with a 1.5× safety factor).
    """
    unstable = 0
    overloaded = 0
    for top in placements:
        top_item = items_by_id[top.item_id]
        supported_area = 0
        if top.position.y_mm == 0:
            supported_area = top.rotated_dimensions.base_area_mm2
        else:
            for bot in placements:
                if bot is top:
                    continue
                if bot.y_max_mm != top.position.y_mm:
                    continue
                ox = max(0, min(top.x_max_mm, bot.x_max_mm) - max(top.position.x_mm, bot.position.x_mm))
                oz = max(0, min(top.z_max_mm, bot.z_max_mm) - max(top.position.z_mm, bot.position.z_mm))
                if ox == 0 or oz == 0:
                    continue
                supported_area += ox * oz
                # pressure applied to the supporting item
                pressure = top_item.pressure_kpa
                bot_item = items_by_id[bot.item_id]
                if pressure > bot_item.crush_strength_kpa * 1.5:
                    overloaded += 1

        if supported_area / top.rotated_dimensions.base_area_mm2 < 0.70:
            unstable += 1

    return unstable, overloaded
