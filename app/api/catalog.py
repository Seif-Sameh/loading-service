"""GET-only catalog endpoints.

These let the frontend populate the cargo / container dropdowns without hard-coding
the catalog on the JS side.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.catalog.loader import imdg_table, list_cargo_presets, list_containers

router = APIRouter(prefix="/api/loading/catalog", tags=["catalog"])


@router.get("/containers")
def get_containers() -> list[dict]:
    """All ISO container types with internal dims, payload, and floor-load rating."""
    return [c.model_dump(mode="json") for c in list_containers()]


@router.get("/cargo-presets")
def get_cargo_presets() -> list[dict]:
    """All cargo presets (raw) — the frontend customises label, weight, etc."""
    return list_cargo_presets()


@router.get("/imdg-segregation")
def get_imdg_segregation() -> dict:
    """The 9×9 IMDG segregation matrix with the distance thresholds used by the solver."""
    t = imdg_table()
    return {
        "classes": list(t.classes),
        "matrix": [list(row) for row in t.matrix],
        "away_from_mm": t.away_from_mm,
        "separated_fore_aft_mm": t.separated_fore_aft_mm,
        "separated_lateral_mm": t.separated_lateral_mm,
    }
