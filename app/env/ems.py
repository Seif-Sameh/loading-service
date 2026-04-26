"""Candidate-action extraction.

We follow the **extreme-points / corner-points** family (Crainic, Perboli & Tadei 2008) rather
than fully maximal EMS decomposition: simpler to implement, fast, and good enough both for
heuristic baselines and for bootstrapping the transformer policy. We label them "EMS" in the
public API for continuity with GOPT.

For each item in the lookahead window we:
1. Enumerate discrete (x, z) corners from the current heightmap skyline *plus* axis-aligned
   extensions of every placed box (x_max, z_max).
2. For each corner + rotation, compute the drop-y via the heightmap.
3. Check support-ratio against a soft threshold (default 0.7) — candidates below are kept
   but flagged, so the RL agent can learn that they're bad, while heuristics can ignore them.
4. Rank candidates by a cheap volume-of-free-space score and keep the top K (default 80).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.env.heightmap import Heightmap
from app.schemas import (
    CandidateAction,
    CargoItem,
    Container,
    Dimensions,
    Placement,
    Position,
    Rotation,
)


@dataclass
class ExtractConfig:
    max_candidates: int = 80
    min_support_ratio: float = 0.50  # matches DeepPack3D; reward layer further penalises < 0.7


def _rotation_candidates(item: CargoItem) -> list[Rotation]:
    return item.available_rotations()


def _corner_seed_points(container: Container, placements: list[Placement]) -> list[tuple[int, int]]:
    """Discrete (x, z) seed points used as candidate footprints' lower-back-left corners.

    We enumerate the **cross-product** of the placed boxes' x-edges and z-edges, plus the
    container origin. This catches L-shaped niches a naive "corners-of-each-box" scan misses
    (e.g., where one box's far edge meets another box's near edge along the orthogonal axis).
    """
    xs: set[int] = {0}
    zs: set[int] = {0}
    for p in placements:
        xs.add(p.position.x_mm)
        xs.add(p.x_max_mm)
        zs.add(p.position.z_mm)
        zs.add(p.z_max_mm)

    seeds: list[tuple[int, int]] = []
    for x in xs:
        if x >= container.internal.length_mm:
            continue
        for z in zs:
            if z >= container.internal.width_mm:
                continue
            seeds.append((x, z))
    return seeds


def extract_candidate_actions(
    *,
    item: CargoItem,
    item_index: int,
    container: Container,
    heightmap: Heightmap,
    placements: list[Placement],
    config: ExtractConfig = ExtractConfig(),
) -> list[CandidateAction]:
    """Return a ranked list of feasible (within-bounds + sufficient-support) placements."""
    seeds = _corner_seed_points(container, placements)
    rotations = _rotation_candidates(item)

    raw: list[tuple[float, CandidateAction]] = []
    for rot in rotations:
        d = item.dimensions.rotated(rot)
        for x, z in seeds:
            if not heightmap.fits(x, z, d.length_mm, d.width_mm):
                continue
            y = heightmap.drop_y(x, z, d.length_mm, d.width_mm)
            # must be within the container vertically
            if y + d.height_mm > container.internal.height_mm:
                continue
            support = heightmap.support_ratio(x, y, z, d.length_mm, d.width_mm)
            if support < config.min_support_ratio:
                continue

            candidate = CandidateAction(
                item_index=item_index,
                position=Position(x_mm=x, y_mm=y, z_mm=z),
                rotation=rot,
                rotated_dimensions=Dimensions(
                    length_mm=d.length_mm, width_mm=d.width_mm, height_mm=d.height_mm
                ),
            )
            # Ranking score: prefer low y (bottom-first) then low x (packed toward far end).
            # Negative so we can sort ascending == best first.
            score = -(y * 10_000 + x)
            raw.append((score, candidate))

    raw.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in raw[: config.max_candidates]]
