"""GA sanity test — small population / few generations so CI stays fast."""
from app.algorithms.ga import GAConfig, GeneticAlgorithm
from app.algorithms.base import solve


def test_ga_plan_then_replay(container_40hc, mixed_bag):
    ga = GeneticAlgorithm(cfg=GAConfig(pop_size=8, generations=3, seed=7))
    ga.prepare(container_40hc, mixed_bag)
    result, events = solve(algorithm=ga, container=container_40hc, items=mixed_bag)
    assert len(result.placements) + len(result.unplaced_item_ids) == len(mixed_bag)
    assert 0.0 < result.kpis.utilization <= 1.0
