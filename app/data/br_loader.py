"""Lazy reader for the parsed BR problems JSON."""
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
    Rotation,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "br"
JSON_PATH = DATA / "br_problems.json"


# ---------------------------------------------------------------------------
# Domain types (lighter than runtime CargoItem — these are *catalog* entries)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BRBoxType:
    type_id: int
    length_mm: int
    width_mm: int
    height_mm: int
    allow_vertical_l: bool
    allow_vertical_w: bool
    allow_vertical_h: bool
    quantity: int


@dataclass(frozen=True)
class BRProblem:
    problem_id: int
    seed_id: int
    source_file: str
    container_lwh_mm: tuple[int, int, int]
    box_types: tuple[BRBoxType, ...]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_all() -> list[BRProblem]:
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"{JSON_PATH} not found. Run `python -m scripts.prepare_datasets` to create it."
        )
    raw = json.loads(JSON_PATH.read_text())
    out: list[BRProblem] = []
    for p in raw["problems"]:
        L, W, H = p["container_cm"]
        types = tuple(
            BRBoxType(
                type_id=b["type_id"],
                # cm → mm so it matches the rest of the codebase
                length_mm=b["length_cm"] * 10,
                width_mm=b["width_cm"] * 10,
                height_mm=b["height_cm"] * 10,
                allow_vertical_l=b["allow_vertical_l"],
                allow_vertical_w=b["allow_vertical_w"],
                allow_vertical_h=b["allow_vertical_h"],
                quantity=b["quantity"],
            )
            for b in p["box_types"]
        )
        out.append(
            BRProblem(
                problem_id=p["problem_id"],
                seed_id=p["seed_id"],
                source_file=p["source_file"],
                container_lwh_mm=(L * 10, W * 10, H * 10),
                box_types=types,
            )
        )
    return out


def list_br_problems() -> list[BRProblem]:
    return list(_load_all())


def load_br_problem(problem_id: int) -> BRProblem:
    for p in _load_all():
        if p.problem_id == problem_id:
            return p
    raise KeyError(f"BR problem id {problem_id!r} not found")


# ---------------------------------------------------------------------------
# Convenience: turn a BR problem into a CargoItem list
# ---------------------------------------------------------------------------


def br_problem_to_items(
    problem: BRProblem,
    *,
    weight_per_box_kg: float = 25.0,
    fragility: FragilityClass = FragilityClass.NORMAL,
    hazmat: HazmatClass = HazmatClass.NONE,
    delivery_stop: int = 0,
) -> list[CargoItem]:
    """Expand a BR problem into a flat list of CargoItem objects.

    BR data has no weights; the caller supplies a default. The :class:`AlexandriaSampler`
    overrides this by drawing real weights from the Wadaboa pool.
    """
    items: list[CargoItem] = []
    counter = 0
    for bt in problem.box_types:
        # If at least one orientation flag is set the item can rotate 90° around the
        # vertical axis only — equivalent to ``allow_all_rotations=False`` and ``this_side_up=False``.
        # If *all* flags are set we let it tumble (allow_all_rotations=True).
        all_rot = bt.allow_vertical_l and bt.allow_vertical_w and bt.allow_vertical_h
        for _ in range(bt.quantity):
            counter += 1
            items.append(
                CargoItem(
                    id=f"br{problem.problem_id}-t{bt.type_id}-{counter:04d}",
                    preset_code=None,
                    label=f"BR{problem.problem_id} type {bt.type_id}",
                    dimensions=Dimensions(
                        length_mm=bt.length_mm,
                        width_mm=bt.width_mm,
                        height_mm=bt.height_mm,
                    ),
                    weight_kg=weight_per_box_kg,
                    fragility=fragility,
                    crush_strength_kpa=120.0,
                    stackable_layers=3,
                    this_side_up=not bt.allow_vertical_h and not all_rot,
                    allow_all_rotations=all_rot,
                    requires_reefer=False,
                    hazmat_class=hazmat,
                    delivery_stop=delivery_stop,
                )
            )
    return items


def br_container_to_isolike(problem: BRProblem) -> Container:
    """Turn a BR container (cm-derived) into a Container schema for runtime use.

    BR1-7 use 587 × 233 × 220 cm which is essentially a 20GP. We construct a synthetic
    Container so its dimensions match the BR instance exactly.
    """
    L, W, H = problem.container_lwh_mm
    return Container(
        code="20GP",  # type: ignore[arg-type]
        display_name=f"BR{problem.problem_id} container",
        internal=Dimensions(length_mm=L, width_mm=W, height_mm=H),
        tare_kg=2100.0,
        payload_kg=28300.0,
        mgw_kg=30480.0,
        floor_load_kg_per_m2=4800.0,
        is_reefer=False,
        is_open_top=False,
        is_flat_rack=False,
    )
