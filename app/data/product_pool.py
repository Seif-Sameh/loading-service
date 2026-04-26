"""Wadaboa product-pool reader.

Loads the 1 M-row parquet on demand. Numpy-only at the read layer so the rest of the
service doesn't depend on pandas at runtime — the parquet reader returns columnar arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "raw"
PARQUET_PATH = DATA / "wadaboa_products.parquet"


@dataclass(frozen=True)
class ProductPool:
    """Columnar arrays of real cargo records."""

    width_mm: np.ndarray   # int32
    depth_mm: np.ndarray   # int32 (used as our "length")
    height_mm: np.ndarray  # int32
    weight_kg: np.ndarray  # int32
    volume_mm3: np.ndarray  # int64

    def __len__(self) -> int:
        return int(self.width_mm.shape[0])

    def filtered(
        self,
        *,
        min_volume_l: float | None = None,
        max_volume_l: float | None = None,
        min_weight_kg: float | None = None,
        max_weight_kg: float | None = None,
        max_dim_mm: int | None = None,
    ) -> "ProductPool":
        """Return a new pool restricted by simple range filters. O(N) once."""
        mask = np.ones(len(self), dtype=bool)
        if min_volume_l is not None:
            mask &= self.volume_mm3 >= min_volume_l * 1_000_000
        if max_volume_l is not None:
            mask &= self.volume_mm3 <= max_volume_l * 1_000_000
        if min_weight_kg is not None:
            mask &= self.weight_kg >= min_weight_kg
        if max_weight_kg is not None:
            mask &= self.weight_kg <= max_weight_kg
        if max_dim_mm is not None:
            mask &= (self.width_mm <= max_dim_mm) & (self.depth_mm <= max_dim_mm) & (self.height_mm <= max_dim_mm)
        return ProductPool(
            width_mm=self.width_mm[mask],
            depth_mm=self.depth_mm[mask],
            height_mm=self.height_mm[mask],
            weight_kg=self.weight_kg[mask],
            volume_mm3=self.volume_mm3[mask],
        )


@lru_cache(maxsize=1)
def load_product_pool() -> ProductPool:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"{PARQUET_PATH} missing. Run `python -m scripts.prepare_datasets "
            f"--wadaboa-pkl PATH/products.pkl` to generate it."
        )
    # Lazy import: pandas/pyarrow only loaded when the pool is actually used.
    import pyarrow.parquet as pq

    table = pq.read_table(PARQUET_PATH)
    cols = {name: table.column(name).to_numpy() for name in table.column_names}
    return ProductPool(
        width_mm=cols["width"].astype(np.int32, copy=False),
        depth_mm=cols["depth"].astype(np.int32, copy=False),
        height_mm=cols["height"].astype(np.int32, copy=False),
        weight_kg=cols["weight"].astype(np.int32, copy=False),
        volume_mm3=cols["volume"].astype(np.int64, copy=False),
    )
