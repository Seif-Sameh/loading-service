"""Shared fixtures for every test module."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the service root is on sys.path when running pytest from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.catalog.loader import get_cargo_preset, get_container
from app.schemas import CargoItem, Container


@pytest.fixture
def container_40hc() -> Container:
    return get_container("40HC")


@pytest.fixture
def container_20gp() -> Container:
    return get_container("20GP")


@pytest.fixture
def eur_pallets_10() -> list[CargoItem]:
    return [get_cargo_preset("eur_pallet_light", item_id=f"p{i:03d}") for i in range(10)]


@pytest.fixture
def mixed_bag() -> list[CargoItem]:
    items: list[CargoItem] = []
    items.append(get_cargo_preset("eur_pallet_heavy", item_id="h1"))
    items.append(get_cargo_preset("us_pallet", item_id="u1"))
    items.append(get_cargo_preset("steel_drum_200l", item_id="d1"))
    items.append(get_cargo_preset("ibc_1000l", item_id="i1"))
    items.append(get_cargo_preset("carton_large", item_id="c1"))
    items.append(get_cargo_preset("bagged_grain_50kg", item_id="b1"))
    return items
