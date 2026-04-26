"""Inference-only wrapper exposing a trained :class:`PackingTransformer` as a
:class:`~app.algorithms.base.PackingAlgorithm`.

The class is import-cheap; it loads PyTorch and the checkpoint lazily so the heuristic /
GA path doesn't pull torch into the main API image.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from app.algorithms.base import PackingAlgorithm
from app.env.packing_env import PackingState


class PPOPackingAgent(PackingAlgorithm):
    """Greedy / sampled inference using a trained Packing Transformer."""

    code = "ppo"
    display_name = "PPO + Packing Transformer"

    def __init__(
        self,
        weights_path: str | os.PathLike | None = None,
        *,
        sample_actions: bool = False,
        device: str = "cpu",
    ) -> None:
        # Lazy torch import.
        import torch

        from app.algorithms.rl.packing_transformer import PackingTransformer
        from app.algorithms.rl.ppo_trainer import PPOTrainer

        if weights_path is None:
            from app.config import settings
            weights_path = settings.rl_weights_path
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"PPO checkpoint missing at {weights_path}. Train via "
                f"`notebooks/03_train_gopt_ppo.ipynb` and save the .pt there."
            )

        self._torch = torch
        self.device = torch.device(device)
        self.model: PackingTransformer = PPOTrainer.load_model(str(weights_path), device=device)
        self.sample_actions = sample_actions

    def select(self, state: PackingState) -> int:
        torch = self._torch
        if not state.candidates:
            return 0

        K = self.model.cfg.ems_dim  # noqa: F841 — for clarity only
        max_K = self.model.cfg.embed_dim and 80  # action space is fixed at 80 in the env

        cont = state.container
        L = cont.internal.length_mm
        W = cont.internal.width_mm
        H = cont.internal.height_mm

        ems = np.zeros((1, 80, 6), dtype=np.float32)
        mask = np.zeros((1, 80), dtype=np.bool_)
        for i, c in enumerate(state.candidates[:80]):
            ems[0, i, 0] = c.position.x_mm / L
            ems[0, i, 1] = c.position.y_mm / H
            ems[0, i, 2] = c.position.z_mm / W
            ems[0, i, 3] = c.rotated_dimensions.length_mm / L
            ems[0, i, 4] = c.rotated_dimensions.width_mm / W
            ems[0, i, 5] = c.rotated_dimensions.height_mm / H
            mask[0, i] = True

        item_arr = np.zeros((1, 2, 3), dtype=np.float32)
        if state.current_item is not None:
            d = state.current_item.dimensions
            item_arr[0, 0] = [d.length_mm / L, d.width_mm / W, d.height_mm / H]
            item_arr[0, 1] = [d.width_mm / L, d.length_mm / W, d.height_mm / H]

        with torch.no_grad():
            t_ems = torch.from_numpy(ems).to(self.device)
            t_item = torch.from_numpy(item_arr).to(self.device)
            t_mask = torch.from_numpy(mask).to(self.device)
            logits, _ = self.model(t_ems, t_item, t_mask)
            n_rot = self.model.cfg.n_rotations
            full_mask = t_mask.unsqueeze(-1).expand(-1, -1, n_rot).reshape(1, -1)
            logits = logits.masked_fill(~full_mask, float("-inf"))
            if self.sample_actions:
                from torch.distributions import Categorical
                action = int(Categorical(logits=logits).sample().item())
            else:
                action = int(logits.argmax(dim=-1).item())

        cand_idx = action // n_rot
        if cand_idx >= len(state.candidates):
            cand_idx = 0
        return cand_idx
