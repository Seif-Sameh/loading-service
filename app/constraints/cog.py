"""Centre-of-gravity bookkeeping.

Keeps a running weighted sum of placed-item centroids so we can read the current CoG
(longitudinal / lateral / vertical) in O(1) without re-iterating the full placement list.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.schemas import Container, Placement


@dataclass
class CoGTracker:
    container: Container
    total_weight_kg: float = 0.0
    _wx_sum: float = 0.0  # weight * x_centroid_mm
    _wy_sum: float = 0.0
    _wz_sum: float = 0.0

    def add(self, placement: Placement, weight_kg: float) -> None:
        cx = placement.position.x_mm + placement.rotated_dimensions.length_mm / 2.0
        cy = placement.position.y_mm + placement.rotated_dimensions.height_mm / 2.0
        cz = placement.position.z_mm + placement.rotated_dimensions.width_mm / 2.0
        self._wx_sum += weight_kg * cx
        self._wy_sum += weight_kg * cy
        self._wz_sum += weight_kg * cz
        self.total_weight_kg += weight_kg

    # ---- derived ----

    @property
    def is_empty(self) -> bool:
        return self.total_weight_kg <= 0.0

    @property
    def longitudinal_fraction(self) -> float:
        """Centre-of-gravity along the length, normalised to [0, 1]. Returns 0.5
        (neutral centre) when the container is empty — so reward terms don't
        penalise a blank state."""
        if self.is_empty:
            return 0.5
        return (self._wx_sum / self.total_weight_kg) / self.container.internal.length_mm

    @property
    def lateral_fraction(self) -> float:
        """Centre-of-gravity across the width, normalised to [0, 1]."""
        if self.is_empty:
            return 0.5
        return (self._wz_sum / self.total_weight_kg) / self.container.internal.width_mm

    @property
    def vertical_fraction(self) -> float:
        """Height of the centre of gravity, normalised to [0, 1]. Returns 0
        when empty because "low" vertical CoG is neutral."""
        if self.is_empty:
            return 0.0
        return (self._wy_sum / self.total_weight_kg) / self.container.internal.height_mm

    @property
    def longitudinal_deviation(self) -> float:
        """Signed deviation from container centre along the length, in [-0.5, 0.5]."""
        return self.longitudinal_fraction - 0.5

    @property
    def lateral_deviation(self) -> float:
        """Signed deviation from container centre across the width, in [-0.5, 0.5]."""
        return self.lateral_fraction - 0.5
