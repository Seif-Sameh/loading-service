"""Packing environment — heightmap, EMS extraction, and the Gymnasium interface.

The environment is intentionally *agnostic* to the policy. Heuristics, the GA, and the
PPO+Transformer agent all consume the same :class:`~app.schemas.CandidateAction` list.
"""
from .ems import ExtractConfig, extract_candidate_actions
from .heightmap import Heightmap
from .packing_env import PackingEnv, PackingState

__all__ = [
    "ExtractConfig",
    "Heightmap",
    "PackingEnv",
    "PackingState",
    "extract_candidate_actions",
]
