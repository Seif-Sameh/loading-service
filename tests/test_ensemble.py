"""EnsembleAgent — best-of-runs over heuristic members."""
from __future__ import annotations


def test_ensemble_matches_or_beats_best_member(container_40hc, mixed_bag):
    """Ensemble must achieve ≥ max member utilisation (it replays the winner's actions)."""
    from app.algorithms import get_algorithm
    from app.algorithms.base import solve
    from app.algorithms.ensemble import EnsembleAgent

    member_codes = ["bl", "extreme_points", "baf", "bssf", "blsf"]
    member_utils = []
    for code in member_codes:
        result, _ = solve(algorithm=get_algorithm(code), container=container_40hc, items=mixed_bag)
        member_utils.append(result.kpis.utilization)
    best_member = max(member_utils)

    ens = EnsembleAgent(ppo_agent=None, heuristic_codes=member_codes)
    result, _ = solve(algorithm=ens, container=container_40hc, items=mixed_bag)
    # Ensemble's util should be at least the best member (within float tolerance).
    assert result.kpis.utilization >= best_member - 1e-6
    # And it should record which member won.
    assert ens.winner_code in member_codes


def test_ensemble_records_action_sequence(container_40hc, eur_pallets_10):
    """attach_env populates the action sequence; select replays it deterministically."""
    from app.algorithms.base import solve
    from app.algorithms.ensemble import EnsembleAgent

    ens = EnsembleAgent(ppo_agent=None)
    result, _ = solve(algorithm=ens, container=container_40hc, items=eur_pallets_10)
    assert len(ens._actions) > 0
    # Replaying the same env should produce the same placements
    assert len(result.placements) == ens._idx
