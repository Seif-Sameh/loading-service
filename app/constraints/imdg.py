"""IMDG hazmat segregation checks.

The matrix is loaded once via :mod:`app.catalog.loader`. At runtime we only need to answer
two questions:

- ``pair_ok(a, b, distance_along_len, distance_across_width)`` — is this *specific pair* of
  hazmat classes allowed in the same container at this distance?
- ``imdg_violations(placements, items)`` — count pairs that violate segregation.
"""
from __future__ import annotations

from collections.abc import Iterable

from app.catalog.loader import imdg_table
from app.schemas import CargoItem, HazmatClass, Placement


def pair_ok(a: HazmatClass, b: HazmatClass, fore_aft_mm: int, lateral_mm: int) -> bool:
    """True iff the two classes can coexist in the same container at the given gaps."""
    tbl = imdg_table()
    code = tbl.segregation_code(a, b)
    if code == 0:
        return True
    if code == 1:  # "away from" — ≥3 m in either axis
        return fore_aft_mm >= tbl.away_from_mm or lateral_mm >= tbl.away_from_mm
    if code == 2:  # "separated from" — ≥6 m fore-aft *or* ≥2.4 m lateral
        return (
            fore_aft_mm >= tbl.separated_fore_aft_mm
            or lateral_mm >= tbl.separated_lateral_mm
        )
    # codes 3 & 4 require different containers entirely
    return False


def _gap_between(a: Placement, b: Placement) -> tuple[int, int]:
    """Minimum gaps along the length (x) and width (z) between two placed boxes."""
    # length axis (x)
    fa_mm = max(a.position.x_mm - b.x_max_mm, b.position.x_mm - a.x_max_mm, 0)
    # width axis (z)
    la_mm = max(a.position.z_mm - b.z_max_mm, b.position.z_mm - a.z_max_mm, 0)
    return fa_mm, la_mm


def imdg_violations(
    placements: Iterable[Placement],
    items_by_id: dict[str, CargoItem],
) -> int:
    """Count hazmat-pair violations across every placed pair."""
    pl = list(placements)
    count = 0
    for i in range(len(pl)):
        item_i = items_by_id[pl[i].item_id]
        if item_i.hazmat_class == HazmatClass.NONE:
            continue
        for j in range(i + 1, len(pl)):
            item_j = items_by_id[pl[j].item_id]
            if item_j.hazmat_class == HazmatClass.NONE:
                continue
            fa, la = _gap_between(pl[i], pl[j])
            if not pair_ok(item_i.hazmat_class, item_j.hazmat_class, fa, la):
                count += 1
    return count
