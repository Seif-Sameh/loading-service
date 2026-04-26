"""Packing algorithms — heuristics, GA, and (later) PPO+Transformer.

All algorithms implement :class:`PackingAlgorithm` so the solver can treat them uniformly.
"""
from .base import PackingAlgorithm, solve
from .ga import GeneticAlgorithm
from .heuristics import (
    BestAreaFit,
    BestLongestSideFit,
    BestShortestSideFit,
    BottomLeft,
    ExtremePoints,
)

ALGORITHM_REGISTRY: dict[str, type[PackingAlgorithm]] = {
    "baf": BestAreaFit,
    "bssf": BestShortestSideFit,
    "blsf": BestLongestSideFit,
    "bl": BottomLeft,
    "extreme_points": ExtremePoints,
    "ga": GeneticAlgorithm,
}


def get_algorithm(code: str, **kwargs) -> PackingAlgorithm:
    """Instantiate an algorithm by registry code.

    The "ppo" code is loaded lazily so this function works in environments without PyTorch.
    """
    if code == "ppo":
        from app.algorithms.rl.ppo_agent import PPOPackingAgent
        return PPOPackingAgent(**kwargs)
    if code not in ALGORITHM_REGISTRY:
        raise KeyError(f"Unknown algorithm: {code!r}. Known: {sorted(ALGORITHM_REGISTRY) + ['ppo']}")
    return ALGORITHM_REGISTRY[code]()


__all__ = [
    "ALGORITHM_REGISTRY",
    "BestAreaFit",
    "BestLongestSideFit",
    "BestShortestSideFit",
    "BottomLeft",
    "ExtremePoints",
    "GeneticAlgorithm",
    "PackingAlgorithm",
    "get_algorithm",
    "solve",
]
