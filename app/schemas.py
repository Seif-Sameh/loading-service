"""Canonical data models used across every layer (catalog, env, algorithms, API).

Dimensions are in **millimetres**, weights in **kilograms**. Keeping fixed units removes a
whole class of bugs. Convert at the API boundary if the frontend prefers metres.
"""
from __future__ import annotations

from enum import Enum, IntEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, field_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ContainerType(str, Enum):
    GP20 = "20GP"
    GP40 = "40GP"
    HC40 = "40HC"
    HC45 = "45HC"
    RF20 = "20RF"
    RF40 = "40RF"
    OT20 = "20OT"
    FR20 = "20FR"


class FragilityClass(IntEnum):
    """Ordinal fragility. 1 = unbreakable (steel), 5 = very fragile (glass/electronics)."""

    UNBREAKABLE = 1
    ROBUST = 2
    NORMAL = 3
    FRAGILE = 4
    VERY_FRAGILE = 5


class HazmatClass(str, Enum):
    """IMDG hazmat classes 1-9. NONE = non-hazardous cargo."""

    NONE = "none"
    C1 = "1"  # Explosives
    C2 = "2"  # Gases
    C3 = "3"  # Flammable liquids
    C4 = "4"  # Flammable solids
    C5 = "5"  # Oxidizers / organic peroxides
    C6 = "6"  # Toxic / infectious
    C7 = "7"  # Radioactive
    C8 = "8"  # Corrosives
    C9 = "9"  # Miscellaneous


class Rotation(IntEnum):
    """Axis-aligned rotations expressed as a permutation of (L, W, H).

    L = length  (x, along container length)
    W = width   (z, across container width)
    H = height  (y, vertical)

    Rotations 0..1 only swap horizontal axes (keeps H up — required for
    `this_side_up` cargo). Rotations 2..5 also tip the item.
    """

    LWH = 0  # (l, w, h) — natural
    WLH = 1  # (w, l, h) — 90 deg around vertical axis
    LHW = 2  # tipped — length along x, height across
    HLW = 3
    WHL = 4
    HWL = 5


ROTATION_PERMUTATIONS: dict[Rotation, tuple[int, int, int]] = {
    # Maps a Rotation to the index permutation (l_idx, w_idx, h_idx) applied to (l, w, h).
    Rotation.LWH: (0, 1, 2),
    Rotation.WLH: (1, 0, 2),
    Rotation.LHW: (0, 2, 1),
    Rotation.HLW: (2, 0, 1),
    Rotation.WHL: (1, 2, 0),
    Rotation.HWL: (2, 1, 0),
}
UPRIGHT_ROTATIONS: tuple[Rotation, Rotation] = (Rotation.LWH, Rotation.WLH)


# ---------------------------------------------------------------------------
# Dimensions & positions
# ---------------------------------------------------------------------------


class Dimensions(BaseModel):
    """Box dimensions in millimetres. L along x, W across z, H up y."""

    model_config = ConfigDict(frozen=True)

    length_mm: PositiveInt
    width_mm: PositiveInt
    height_mm: PositiveInt

    @property
    def volume_mm3(self) -> int:
        return self.length_mm * self.width_mm * self.height_mm

    @property
    def base_area_mm2(self) -> int:
        return self.length_mm * self.width_mm

    def rotated(self, rotation: Rotation) -> "Dimensions":
        perm = ROTATION_PERMUTATIONS[rotation]
        src = (self.length_mm, self.width_mm, self.height_mm)
        return Dimensions(
            length_mm=src[perm[0]],
            width_mm=src[perm[1]],
            height_mm=src[perm[2]],
        )


class Position(BaseModel):
    """Lower-back-left corner of a placed box, in container-local mm coords."""

    model_config = ConfigDict(frozen=True)

    x_mm: NonNegativeInt  # along container length
    y_mm: NonNegativeInt  # vertical
    z_mm: NonNegativeInt  # across container width


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


class Container(BaseModel):
    """ISO shipping container with real-world capacity limits."""

    code: ContainerType
    display_name: str
    internal: Dimensions
    tare_kg: float = Field(gt=0)
    payload_kg: float = Field(gt=0)
    mgw_kg: float = Field(gt=0)
    floor_load_kg_per_m2: float = Field(gt=0)
    is_reefer: bool = False
    is_open_top: bool = False
    is_flat_rack: bool = False

    @property
    def floor_area_m2(self) -> float:
        return (self.internal.length_mm * self.internal.width_mm) / 1_000_000.0

    @property
    def volume_m3(self) -> float:
        return self.internal.volume_mm3 / 1_000_000_000.0


# ---------------------------------------------------------------------------
# Cargo
# ---------------------------------------------------------------------------


class CargoItem(BaseModel):
    """One piece of cargo to be packed."""

    id: str
    preset_code: str | None = None
    label: str | None = None

    dimensions: Dimensions
    weight_kg: float = Field(gt=0)

    # Physical properties
    fragility: FragilityClass = FragilityClass.NORMAL
    crush_strength_kpa: float = Field(default=100.0, gt=0)
    stackable_layers: PositiveInt = 3

    # Constraints
    this_side_up: bool = False
    allow_all_rotations: bool = False
    requires_reefer: bool = False
    hazmat_class: HazmatClass = HazmatClass.NONE

    # Logistics
    delivery_stop: NonNegativeInt = 0  # 0 = single-drop; larger = later stop

    @field_validator("stackable_layers")
    @classmethod
    def _cap_stack(cls, v: int) -> int:
        return min(v, 10)

    @property
    def base_area_m2(self) -> float:
        return self.dimensions.base_area_mm2 / 1_000_000.0

    @property
    def pressure_kpa(self) -> float:
        """Floor/item pressure produced by this cargo when resting on its current base."""
        # weight(kg) * g(≈9.81 m/s²) / base_area(m²) in pascals, then /1000 for kPa
        return (self.weight_kg * 9.81) / self.base_area_m2 / 1000.0

    def available_rotations(self) -> list[Rotation]:
        if self.allow_all_rotations:
            return list(Rotation)
        if self.this_side_up:
            return list(UPRIGHT_ROTATIONS)
        return list(UPRIGHT_ROTATIONS)  # default = vertical-preserving only


# ---------------------------------------------------------------------------
# Placements and state
# ---------------------------------------------------------------------------


class Placement(BaseModel):
    """A cargo item placed inside a container."""

    item_id: str
    position: Position
    rotation: Rotation
    rotated_dimensions: Dimensions

    @property
    def x_max_mm(self) -> int:
        return self.position.x_mm + self.rotated_dimensions.length_mm

    @property
    def y_max_mm(self) -> int:
        return self.position.y_mm + self.rotated_dimensions.height_mm

    @property
    def z_max_mm(self) -> int:
        return self.position.z_mm + self.rotated_dimensions.width_mm


class CandidateAction(BaseModel):
    """One feasible placement proposal, produced by the environment and scored by algorithms."""

    item_index: int  # index into the current lookahead window
    position: Position
    rotation: Rotation
    rotated_dimensions: Dimensions


class KPIs(BaseModel):
    """Quality metrics for a running or finished solve."""

    utilization: float = 0.0  # volume packed / container volume
    weight_used: float = 0.0  # placed weight / payload
    cog_long_dev: float = 0.0  # signed deviation from container centre, normalised [-0.5, 0.5]
    cog_lat_dev: float = 0.0
    cog_vert_frac: float = 0.0  # vertical CoG / container height [0, 1]
    unstable_count: NonNegativeInt = 0
    overloaded_count: NonNegativeInt = 0
    imdg_violation_count: NonNegativeInt = 0
    lifo_violation_count: NonNegativeInt = 0
    stack_violation_count: NonNegativeInt = 0


class SolveResult(BaseModel):
    """Completion payload after running an algorithm end-to-end."""

    algorithm: str
    container_code: ContainerType
    placements: list[Placement]
    unplaced_item_ids: list[str]
    kpis: KPIs
    elapsed_ms: float


# ---------------------------------------------------------------------------
# API DTOs
# ---------------------------------------------------------------------------


class SolveRequest(BaseModel):
    container_code: ContainerType
    items: list[CargoItem]
    algorithm: Literal["baf", "bssf", "blsf", "bl", "extreme_points", "ga", "ppo", "legacy_dqn"] = (
        "extreme_points"
    )
    seed: int | None = None
