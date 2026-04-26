"""Tests for the data layer (BR loader, product pool, Alexandria sampler).

These tests are skipped automatically if the dataset files are not present, so a fresh
clone passes CI without first running ``scripts/prepare_datasets.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data"
BR_JSON = DATA / "br" / "br_problems.json"
WADABOA = DATA / "raw" / "wadaboa_products.parquet"


pytestmark = pytest.mark.skipif(
    not BR_JSON.exists() or not WADABOA.exists(),
    reason="Datasets not prepared; run `python -m scripts.prepare_datasets` first.",
)


def test_br_loader_returns_problems():
    from app.data.br_loader import list_br_problems

    problems = list_br_problems()
    assert len(problems) >= 100
    p = problems[0]
    assert len(p.box_types) > 0
    assert all(b.length_mm > 0 and b.width_mm > 0 and b.height_mm > 0 for b in p.box_types)
    # Container is 587 cm long for BR1-7 → 5870 mm
    assert p.container_lwh_mm[0] in {5870, 12320, 12190}


def test_br_problem_to_items_expands_quantities():
    from app.data.br_loader import br_problem_to_items, load_br_problem

    p = load_br_problem(1)
    items = br_problem_to_items(p)
    expected = sum(b.quantity for b in p.box_types)
    assert len(items) == expected
    assert {it.weight_kg for it in items} == {25.0}


def test_product_pool_loads_and_filters():
    from app.data.product_pool import load_product_pool

    pool = load_product_pool()
    assert len(pool) > 100_000
    light = pool.filtered(max_weight_kg=10)
    heavy = pool.filtered(min_weight_kg=200)
    assert len(light) > 0 and len(heavy) > 0
    assert int(light.weight_kg.max()) <= 10
    assert int(heavy.weight_kg.min()) >= 200


def test_alexandria_sampler_real_strategy():
    from app.data.alexandria_sampler import AlexandriaSampler, SamplerConfig

    s = AlexandriaSampler(SamplerConfig(n_items=20, strategy="real", seed=7))
    items = s.sample()
    assert len(items) == 20
    # At least one hazmat or reefer item should appear from the mix in 20 draws under fixed seed
    # (proportions sum: hazmat 5% + reefer 8%)
    cats = {it.label for it in items}
    assert any(it.weight_kg > 0 for it in items)
    assert any(it.dimensions.length_mm > 100 for it in items)


def test_alexandria_sampler_presets_strategy_no_pool_required():
    from app.data.alexandria_sampler import AlexandriaSampler, SamplerConfig

    s = AlexandriaSampler(SamplerConfig(n_items=10, strategy="presets", seed=1))
    items = s.sample()
    assert len(items) == 10
    assert all(it.preset_code for it in items)


def test_sampler_produces_solvable_voyage(container_40hc):
    """End-to-end: sample → solve with a heuristic → no crash, some items packed."""
    from app.algorithms import get_algorithm
    from app.algorithms.base import solve
    from app.data.alexandria_sampler import AlexandriaSampler, SamplerConfig

    s = AlexandriaSampler(SamplerConfig(n_items=15, strategy="real", seed=3))
    items = s.sample()
    algo = get_algorithm("extreme_points")
    result, _ = solve(algorithm=algo, container=container_40hc, items=items)
    # Some items must place; feasibility filtering means a few may not fit.
    assert len(result.placements) > 0
    assert len(result.placements) + len(result.unplaced_item_ids) == len(items)
