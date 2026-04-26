"""Reinforcement-learning agents.

The module is split so that *importing* the package does not pull in PyTorch — only when
the user explicitly asks for the PPO agent. Heuristic / GA users keep a slim dependency tree.
"""
from .packing_transformer import PackingTransformer, PackingTransformerConfig

# PPO trainer / agent are pulled in lazily; see ppo_agent.py.

__all__ = ["PackingTransformer", "PackingTransformerConfig"]
