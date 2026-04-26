"""Hard-constraint feasibility mask.

Every candidate placement produced by the environment is first filtered through these checks.
An algorithm can then choose freely among feasible candidates without worrying about physical
violations — they're already impossible to pick.

The checks here are O(1) per placement given precomputed state (except IMDG which is O(n)
against already-placed items).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.constraints.imdg import pair_ok
from app.schemas import (
    CandidateAction,
    CargoItem,
    Container,
    HazmatClass,
    Placement,
    UPRIGHT_ROTATIONS,
)


# Container floor load is a hard limit. Soft-tier reward penalises near-limit pressure separately.
# Keeping it hard (rather than soft) matches the hansatic guidance that exceeding it is unsafe.


@dataclass
class FeasibilityMask:
    """A boolean mask over candidate actions (True = feasible)."""

    candidates: list[CandidateAction]
    feasible: list[bool]

    def filter_feasible(self) -> list[CandidateAction]:
        return [c for c, ok in zip(self.candidates, self.feasible, strict=True) if ok]


def is_placement_feasible(
    candidate: CandidateAction,
    item: CargoItem,
    container: Container,
    placed: list[Placement],
    items_by_id: dict[str, CargoItem],
    *,
    current_total_weight_kg: float,
) -> bool:
    """Check every hard constraint for a single candidate placement."""
    # H2: within-bounds
    if candidate.position.x_mm + candidate.rotated_dimensions.length_mm > container.internal.length_mm:
        return False
    if candidate.position.z_mm + candidate.rotated_dimensions.width_mm > container.internal.width_mm:
        return False
    if candidate.position.y_mm + candidate.rotated_dimensions.height_mm > container.internal.height_mm:
        return False

    # H5: orientation lock — only upright rotations allowed if this_side_up
    if item.this_side_up and candidate.rotation not in UPRIGHT_ROTATIONS:
        return False

    # H6: reefer-only
    if item.requires_reefer and not container.is_reefer:
        return False

    # H3: payload not exceeded
    if current_total_weight_kg + item.weight_kg > container.payload_kg:
        return False

    # H4: floor load rating (only checked for items resting on the floor)
    if candidate.position.y_mm == 0:
        base_m2 = candidate.rotated_dimensions.base_area_mm2 / 1_000_000.0
        pressure_kg_per_m2 = item.weight_kg / base_m2 if base_m2 > 0 else 0.0
        if pressure_kg_per_m2 > container.floor_load_kg_per_m2:
            return False

    # H7: IMDG segregation — against items already placed
    if item.hazmat_class != HazmatClass.NONE:
        cand_x_max = candidate.position.x_mm + candidate.rotated_dimensions.length_mm
        cand_z_max = candidate.position.z_mm + candidate.rotated_dimensions.width_mm
        for p in placed:
            other = items_by_id[p.item_id]
            if other.hazmat_class == HazmatClass.NONE:
                continue
            fa = max(candidate.position.x_mm - p.x_max_mm, p.position.x_mm - cand_x_max, 0)
            la = max(candidate.position.z_mm - p.z_max_mm, p.position.z_mm - cand_z_max, 0)
            if not pair_ok(item.hazmat_class, other.hazmat_class, fa, la):
                return False

    return True


def build_feasibility_mask(
    candidates: list[CandidateAction],
    item: CargoItem,
    container: Container,
    placed: list[Placement],
    items_by_id: dict[str, CargoItem],
    *,
    current_total_weight_kg: float,
) -> FeasibilityMask:
    """Produce a feasibility mask over a list of candidate placements."""
    feasible = [
        is_placement_feasible(
            c,
            item,
            container,
            placed,
            items_by_id,
            current_total_weight_kg=current_total_weight_kg,
        )
        for c in candidates
    ]
    return FeasibilityMask(candidates=candidates, feasible=feasible)
