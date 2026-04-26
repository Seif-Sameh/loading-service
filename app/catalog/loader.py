"""Reads the JSON catalogs and hands them back as validated Pydantic objects.

The files are loaded once on import and memoised for the life of the process.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.schemas import (
    CargoItem,
    Container,
    Dimensions,
    FragilityClass,
    HazmatClass,
)

CATALOG_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_container_file() -> list[Container]:
    data = json.loads((CATALOG_DIR / "containers.json").read_text(encoding="utf-8"))
    return [Container.model_validate(entry) for entry in data]


def list_containers() -> list[Container]:
    """Return every ISO container in the catalog."""
    return list(_load_container_file())


def get_container(code: str) -> Container:
    """Look up a container by its ISO code (e.g. ``"40HC"``)."""
    for c in _load_container_file():
        if c.code.value == code:
            return c
    raise KeyError(f"Unknown container code: {code!r}")


# ---------------------------------------------------------------------------
# Cargo presets
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_cargo_presets() -> dict[str, dict]:
    data = json.loads((CATALOG_DIR / "cargo_presets.json").read_text(encoding="utf-8"))
    return {entry["code"]: entry for entry in data}


def list_cargo_presets() -> list[dict]:
    """Return every cargo preset (raw dicts — use :func:`get_cargo_preset` to materialise)."""
    return list(_load_cargo_presets().values())


def get_cargo_preset(code: str, item_id: str, *, overrides: dict | None = None) -> CargoItem:
    """Instantiate a :class:`CargoItem` from a preset code.

    Parameters
    ----------
    code:
        Key in ``cargo_presets.json`` (e.g. ``"eur_pallet_heavy"``).
    item_id:
        Unique id for the resulting cargo item. The caller is responsible for uniqueness.
    overrides:
        Optional dict merged on top of the preset — used when a user tweaks weight or stop.
    """
    presets = _load_cargo_presets()
    if code not in presets:
        raise KeyError(f"Unknown cargo preset: {code!r}")

    spec = dict(presets[code])
    if overrides:
        spec.update(overrides)

    return CargoItem(
        id=item_id,
        preset_code=code,
        label=spec.get("label"),
        dimensions=Dimensions(**spec["dimensions"]),
        weight_kg=spec["weight_kg"],
        fragility=FragilityClass(spec.get("fragility", 3)),
        crush_strength_kpa=spec.get("crush_strength_kpa", 100.0),
        stackable_layers=spec.get("stackable_layers", 3),
        this_side_up=spec.get("this_side_up", False),
        allow_all_rotations=spec.get("allow_all_rotations", False),
        requires_reefer=spec.get("requires_reefer", False),
        hazmat_class=HazmatClass(spec.get("hazmat_class", "none")),
        delivery_stop=spec.get("delivery_stop", 0),
    )


# ---------------------------------------------------------------------------
# IMDG segregation table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IMDGTable:
    """Typed wrapper around ``imdg_segregation.json``.

    Segregation codes:
    - 0: no restriction
    - 1: "away from" — minimum 3 m
    - 2: "separated from" — 6 m fore-aft or 2.4 m lateral
    - 3: "separated by a complete compartment" — different container required
    - 4: "separated longitudinally by an intervening complete compartment" — strictest
    """

    classes: tuple[str, ...]
    matrix: tuple[tuple[int, ...], ...]
    away_from_mm: int
    separated_fore_aft_mm: int
    separated_lateral_mm: int

    def segregation_code(self, a: HazmatClass, b: HazmatClass) -> int:
        i = self.classes.index(a.value)
        j = self.classes.index(b.value)
        return self.matrix[i][j]


@lru_cache(maxsize=1)
def imdg_table() -> IMDGTable:
    raw = json.loads((CATALOG_DIR / "imdg_segregation.json").read_text(encoding="utf-8"))
    return IMDGTable(
        classes=tuple(raw["classes"]),
        matrix=tuple(tuple(row) for row in raw["matrix"]),
        away_from_mm=int(raw["distance_mm_for_away_from"]),
        separated_fore_aft_mm=int(raw["distance_mm_fore_aft_for_separated_from"]),
        separated_lateral_mm=int(raw["distance_mm_lateral_for_separated_from"]),
    )
