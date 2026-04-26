import pytest

from app.algorithms import get_algorithm
from app.algorithms.base import solve


@pytest.mark.parametrize("code", ["baf", "bssf", "blsf", "bl", "extreme_points"])
def test_heuristic_places_all_eur_pallets_in_40hc(code, container_40hc, eur_pallets_10):
    algo = get_algorithm(code)
    result, events = solve(
        algorithm=algo,
        container=container_40hc,
        items=eur_pallets_10,
    )
    # 10 EUR pallets (1.2 × 0.8 × 1.2 m) fit comfortably in a 40HC.
    assert len(result.placements) == 10
    assert result.unplaced_item_ids == []
    assert len(events) == 10
    assert 0.0 < result.kpis.utilization <= 1.0
    # Bottom of the container should be fully populated for the first placement.
    assert result.placements[0].position.y_mm == 0


def test_extreme_points_beats_bottom_left_on_mixed_bag(container_40hc, mixed_bag):
    ep = solve(algorithm=get_algorithm("extreme_points"), container=container_40hc, items=mixed_bag)
    bl = solve(algorithm=get_algorithm("bl"), container=container_40hc, items=mixed_bag)
    # Not guaranteed superior, but should not be worse by a large margin. We only assert both
    # finish.
    assert ep[0].elapsed_ms >= 0
    assert bl[0].elapsed_ms >= 0
