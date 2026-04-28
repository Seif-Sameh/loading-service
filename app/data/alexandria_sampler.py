"""Sampler that draws realistic Alexandria-Port voyages.

Two strategies are wired in, controlled by :class:`SamplerConfig`:

- ``"real"``  — for each commodity category, draw `(w, d, h, kg)` from the Wadaboa real-
  product pool restricted by that category's filter (volume / weight bands). Categories are
  chosen multinomially according to ``alexandria_cargo_mix.json``.
- ``"presets"`` — fall back to the curated catalog presets (good for quick demos /
  deterministic sanity tests).

Each sampled :class:`CargoItem` carries the category's hazmat class, fragility, and
``requires_reefer`` flag — so the constraint layer immediately becomes meaningful.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from app.catalog.loader import get_cargo_preset
from app.data.product_pool import ProductPool, load_product_pool
from app.schemas import (
    CargoItem,
    Dimensions,
    FragilityClass,
    HazmatClass,
)

CARGO_MIX_PATH = Path(__file__).resolve().parents[2] / "data" / "alexandria_cargo_mix.json"


@lru_cache(maxsize=1)
def _load_mix() -> list[dict]:
    return json.loads(CARGO_MIX_PATH.read_text())["categories"]


@dataclass
class SamplerConfig:
    """Knobs for :class:`AlexandriaSampler`.

    Defaults are tuned so a 40HC container actually fills up:
    - n_items=200 (many items per voyage)
    - strategy="mixed" — 60 % real Wadaboa pool + 40 % catalog presets so each voyage
      contains a realistic blend of small parcels and bigger industrial items
      (pallets, drums, IBCs).

    The pure "real" strategy alone leaves 40HC voyages mostly empty (median Wadaboa item
    is ~13 L, so 200 items only fill ~3 % of a 76,000 L container). The mix gives
    heuristics ~60-75 % utilisation on realistic-looking voyages.
    """

    n_items: int = 200
    strategy: Literal["real", "presets", "mixed"] = "mixed"
    real_share: float = 0.60  # used by "mixed": fraction of items drawn from Wadaboa pool
    seed: int | None = 0
    cap_total_weight_kg: float | None = None


class AlexandriaSampler:
    """Stateful sampler. Materialise once per voyage."""

    def __init__(self, cfg: SamplerConfig | None = None) -> None:
        self.cfg = cfg or SamplerConfig()
        self.rng = random.Random(self.cfg.seed)
        self.np_rng = np.random.default_rng(self.cfg.seed)
        self._mix = _load_mix()
        self._pool: ProductPool | None = None
        self._filtered_pools: dict[str, ProductPool] = {}

    # ----- public API -----

    def sample(self) -> list[CargoItem]:
        if self.cfg.strategy == "presets":
            return self._sample_from_presets()
        if self.cfg.strategy == "real":
            return self._sample_from_real_pool()
        return self._sample_mixed()

    def _sample_mixed(self) -> list[CargoItem]:
        """Mix real Wadaboa-pool items with catalog presets per item.

        The pool gives realistic small-parcel variety; presets contribute the bigger
        items (pallets, IBCs, drums, machinery) that actually fill containers.
        """
        if self._pool is None:
            self._pool = load_product_pool()
        items: list[CargoItem] = []
        for i in range(self.cfg.n_items):
            cat = self._draw_category()
            use_real = self.rng.random() < self.cfg.real_share
            if use_real:
                pool = self._filtered_pool(cat)
                if len(pool) == 0:
                    use_real = False
            if use_real:
                row = int(self.np_rng.integers(0, len(pool)))
                d = Dimensions(
                    length_mm=int(pool.depth_mm[row]),
                    width_mm=int(pool.width_mm[row]),
                    height_mm=int(pool.height_mm[row]),
                )
                items.append(
                    CargoItem(
                        id=f"alex-{i:04d}",
                        preset_code=None,
                        label=cat["name"],
                        dimensions=d,
                        weight_kg=float(pool.weight_kg[row]),
                        fragility=FragilityClass(cat.get("fragility", 3)),
                        crush_strength_kpa=120.0,
                        stackable_layers=3,
                        this_side_up=cat.get("hazmat_class", "none") != "none",
                        allow_all_rotations=False,
                        requires_reefer=cat.get("requires_reefer", False),
                        hazmat_class=HazmatClass(cat.get("hazmat_class", "none")),
                        delivery_stop=0,
                    )
                )
            else:
                preset_code = self.rng.choice(cat["presets"])
                items.append(get_cargo_preset(preset_code, item_id=f"alex-{i:04d}"))
        return items

    # ----- presets path -----

    def _sample_from_presets(self) -> list[CargoItem]:
        items: list[CargoItem] = []
        for i in range(self.cfg.n_items):
            cat = self._draw_category()
            preset_code = self.rng.choice(cat["presets"])
            it = get_cargo_preset(preset_code, item_id=f"alex-{i:04d}")
            items.append(it)
        return items

    # ----- real-pool path -----

    def _sample_from_real_pool(self) -> list[CargoItem]:
        if self._pool is None:
            self._pool = load_product_pool()
        items: list[CargoItem] = []
        for i in range(self.cfg.n_items):
            cat = self._draw_category()
            pool = self._filtered_pool(cat)
            if len(pool) == 0:
                # Fall back to preset if filter is too strict
                preset_code = self.rng.choice(cat["presets"])
                items.append(get_cargo_preset(preset_code, item_id=f"alex-{i:04d}"))
                continue
            row = int(self.np_rng.integers(0, len(pool)))
            d = Dimensions(
                length_mm=int(pool.depth_mm[row]),
                width_mm=int(pool.width_mm[row]),
                height_mm=int(pool.height_mm[row]),
            )
            items.append(
                CargoItem(
                    id=f"alex-{i:04d}",
                    preset_code=None,
                    label=cat["name"],
                    dimensions=d,
                    weight_kg=float(pool.weight_kg[row]),
                    fragility=FragilityClass(cat.get("fragility", 3)),
                    crush_strength_kpa=120.0,
                    stackable_layers=3,
                    this_side_up=cat.get("hazmat_class", "none") != "none",
                    allow_all_rotations=False,
                    requires_reefer=cat.get("requires_reefer", False),
                    hazmat_class=HazmatClass(cat.get("hazmat_class", "none")),
                    delivery_stop=0,
                )
            )
        return items

    # ----- helpers -----

    def _draw_category(self) -> dict:
        weights = [c["proportion"] for c in self._mix]
        names = [c["name"] for c in self._mix]
        chosen = self.np_rng.choice(len(names), p=np.array(weights) / sum(weights))
        return self._mix[int(chosen)]

    def _filtered_pool(self, cat: dict) -> ProductPool:
        cache_key = cat["name"]
        if cache_key in self._filtered_pools:
            return self._filtered_pools[cache_key]
        f = cat.get("real_filter", {}) or {}
        assert self._pool is not None
        pool = self._pool.filtered(
            min_volume_l=f.get("min_volume_l"),
            max_volume_l=f.get("max_volume_l"),
            min_weight_kg=f.get("min_weight_kg"),
            max_weight_kg=f.get("max_weight_kg"),
            max_dim_mm=1500,
        )
        self._filtered_pools[cache_key] = pool
        return pool
