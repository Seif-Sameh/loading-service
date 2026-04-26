"""Discrete 2-D heightmap of the container floor.

The heightmap H[i, k] stores the top-of-stack y-coordinate (mm) at cell (i, k), where
``i = x // resolution`` and ``k = z // resolution``. It's the standard compact state used by
DeepPack3D and by GOPT's corner-point search.

Why mm granularity with a user-tunable resolution:
- The real bin is 12000 × 2352 mm. A 50 mm grid gives 240 × 48 = 11 520 cells — fast.
- A finer grid (10 mm) triples the footprint; coarser (100 mm) halves it but loses precision
  for small cartons.

Default 50 mm = 5 cm is a sensible trade-off: most real cargo is a multiple of this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.schemas import Container, Position


@dataclass
class Heightmap:
    container: Container
    resolution_mm: int = 10
    _grid: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.resolution_mm <= 0:
            raise ValueError("resolution_mm must be positive")
        self._grid = np.zeros(
            (
                self.container.internal.length_mm // self.resolution_mm + 1,
                self.container.internal.width_mm // self.resolution_mm + 1,
            ),
            dtype=np.int32,
        )

    # ---- conversions ----

    def _xz_to_cell(self, x_mm: int, z_mm: int) -> tuple[int, int]:
        return x_mm // self.resolution_mm, z_mm // self.resolution_mm

    def _footprint_cells(self, x_mm: int, z_mm: int, l_mm: int, w_mm: int) -> tuple[slice, slice]:
        i0, k0 = self._xz_to_cell(x_mm, z_mm)
        # Ceil-divide so the footprint fully covers the item even when non-aligned.
        i1 = -(-(x_mm + l_mm) // self.resolution_mm)
        k1 = -(-(z_mm + w_mm) // self.resolution_mm)
        return slice(i0, i1), slice(k0, k1)

    # ---- queries ----

    def drop_y(self, x_mm: int, z_mm: int, l_mm: int, w_mm: int) -> int:
        """Return the y at which an item of footprint l×w would come to rest at (x, z)."""
        ri, rk = self._footprint_cells(x_mm, z_mm, l_mm, w_mm)
        patch = self._grid[ri, rk]
        return int(patch.max()) if patch.size else 0

    def support_ratio(self, x_mm: int, y_mm: int, z_mm: int, l_mm: int, w_mm: int) -> float:
        """Fraction of the item's base at cells whose top equals the drop height."""
        ri, rk = self._footprint_cells(x_mm, z_mm, l_mm, w_mm)
        patch = self._grid[ri, rk]
        if patch.size == 0:
            return 0.0
        if y_mm == 0:
            return 1.0  # floor always supports 100 %
        return float(np.sum(patch == y_mm)) / float(patch.size)

    def fits(self, x_mm: int, z_mm: int, l_mm: int, w_mm: int) -> bool:
        return (
            x_mm + l_mm <= self.container.internal.length_mm
            and z_mm + w_mm <= self.container.internal.width_mm
        )

    # ---- mutation ----

    def place(self, position: Position, l_mm: int, w_mm: int, h_mm: int) -> None:
        ri, rk = self._footprint_cells(position.x_mm, position.z_mm, l_mm, w_mm)
        top = position.y_mm + h_mm
        self._grid[ri, rk] = top

    # ---- serialisation / debug ----

    def as_numpy(self) -> np.ndarray:
        return self._grid.copy()

    def skyline_points(self) -> list[tuple[int, int, int]]:
        """Return distinct (x, z, y) points where the heightmap changes — candidate corners."""
        res = self.resolution_mm
        grid = self._grid
        pts: set[tuple[int, int, int]] = set()
        rows, cols = grid.shape
        for i in range(rows):
            for k in range(cols):
                y = int(grid[i, k])
                pts.add((i * res, k * res, y))
        return sorted(pts)
