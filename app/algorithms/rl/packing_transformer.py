"""Packing Transformer — actor-critic backbone for online 3D bin packing.

This is the GOPT-style architecture (Xiong et al. RA-L 2024) ported to fit our environment.
Inputs come straight from :py:meth:`app.env.packing_env.PackingEnv._obs`:

- ``ems``  : ``(B, K, 6)`` candidate placements normalised to [0, 1].
- ``item`` : ``(B, 2, 3)`` two upright rotations of the next item.
- ``mask`` : ``(B, K)`` 1 where the candidate is feasible.

Output: ``(B, 2K)`` action logits (each candidate × {default-orient, swap-orient}).

We deliberately keep the network compact so it trains fast on a single Colab T4. The
attention configuration follows GOPT's *small* setting (3 blocks, 128-D embeddings, 4
heads). Empirically this hits ~75 % space utilisation on 10×10×10 synthetic data after
~6 hours on an RTX 3090; on a T4 expect ~24 h to convergence.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PackingTransformerConfig:
    embed_dim: int = 192          # bumped from 128 for Option-B sizing
    n_heads: int = 4
    n_encoder_blocks: int = 4     # bumped from 3
    mlp_hidden: int = 384         # bumped from 256
    dropout: float = 0.0
    n_rotations: int = 2          # upright-only: LWH and WLH
    ems_dim: int = 6              # (x, y, z, l, w, h) normalised
    lookahead: int = 5            # number of items the encoder attends to (current + 4 future)


class _MultiHeadCrossAttention(nn.Module):
    """Simple multi-head attention block with residual + LayerNorm.

    Used both for self-attention (when ``q is k is v``) and cross-attention (when ``q``
    differs from ``k/v``)."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        return self.norm(q + attn_out)


class _MLPBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.layers(x))


class _PackingEncoderBlock(nn.Module):
    """One encoder block: EMS self-attn → item self-attn → bidirectional cross-attn → MLP.

    Both branches accept a key-padding mask so unused slots (zero-padded EMSs and
    zero-padded lookahead items) are excluded from attention.
    """

    def __init__(self, cfg: PackingTransformerConfig) -> None:
        super().__init__()
        self.ems_self = _MultiHeadCrossAttention(cfg.embed_dim, cfg.n_heads, cfg.dropout)
        self.item_self = _MultiHeadCrossAttention(cfg.embed_dim, cfg.n_heads, cfg.dropout)
        self.ems_to_item = _MultiHeadCrossAttention(cfg.embed_dim, cfg.n_heads, cfg.dropout)
        self.item_to_ems = _MultiHeadCrossAttention(cfg.embed_dim, cfg.n_heads, cfg.dropout)
        self.ems_mlp = _MLPBlock(cfg.embed_dim, cfg.mlp_hidden, cfg.dropout)
        self.item_mlp = _MLPBlock(cfg.embed_dim, cfg.mlp_hidden, cfg.dropout)

    def forward(
        self,
        ems: torch.Tensor,
        item: torch.Tensor,
        ems_pad_mask: torch.Tensor,
        item_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ems = self.ems_self(ems, ems, ems, key_padding_mask=ems_pad_mask)
        item = self.item_self(item, item, item, key_padding_mask=item_pad_mask)
        new_ems = self.ems_to_item(ems, item, item, key_padding_mask=item_pad_mask)
        new_item = self.item_to_ems(item, ems, ems, key_padding_mask=ems_pad_mask)
        new_ems = self.ems_mlp(new_ems)
        new_item = self.item_mlp(new_item)
        return new_ems, new_item


class PackingTransformer(nn.Module):
    """Actor + critic with a shared encoder over (EMS, item) tokens."""

    def __init__(self, cfg: PackingTransformerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PackingTransformerConfig()
        self.ems_proj = nn.Linear(self.cfg.ems_dim, self.cfg.embed_dim)
        self.item_proj = nn.Linear(3, self.cfg.embed_dim)
        self.blocks = nn.ModuleList(
            [_PackingEncoderBlock(self.cfg) for _ in range(self.cfg.n_encoder_blocks)]
        )
        # Actor head: per-EMS × per-rotation logit
        self.actor_head = nn.Linear(self.cfg.embed_dim * 2, self.cfg.n_rotations)
        # Critic head: pooled state value
        self.critic_head = nn.Sequential(
            nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim),
            nn.GELU(),
            nn.Linear(self.cfg.embed_dim, 1),
        )

    # ----- public -----

    def forward(
        self,
        ems: torch.Tensor,            # (B, K, 6) float in [0, 1]
        items: torch.Tensor,          # (B, L, R, 3) float in [0, 1]
        mask: torch.Tensor,           # (B, K) bool, True = valid EMS
        items_mask: torch.Tensor | None = None,  # (B, L) bool, True = real item (not padding)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_logits, state_value).

        ``action_logits`` has shape ``(B, K * n_rotations)`` — one logit per
        (EMS index, rotation) pair for the **current** item, masked to -inf where the
        EMS slot is padding.

        The lookahead items beyond slot 0 are context for the encoder's cross-attention;
        they don't add to the action space.
        """
        B, K, _ = ems.shape
        L = items.shape[1]
        R = items.shape[2]
        if items_mask is None:
            items_mask = torch.ones(B, L, dtype=torch.bool, device=items.device)

        ems_pad = ~mask
        # Each lookahead-item × rotation becomes a separate token: (B, L*R, 3) → embed.
        item_seq = items.reshape(B, L * R, 3)
        # Tile the items_mask across rotations so both rotations of one item share its mask.
        item_pad = ~items_mask.unsqueeze(-1).expand(-1, -1, R).reshape(B, L * R)

        e = self.ems_proj(ems)
        i = self.item_proj(item_seq)
        for block in self.blocks:
            e, i = block(e, i, ems_pad, item_pad)

        # Action head uses ONLY the current-item rotations (slots [0, R)) of the item sequence.
        current_item_emb = i[:, :R, :]                       # (B, R, embed)
        i_exp = current_item_emb.unsqueeze(1).expand(-1, K, -1, -1)        # (B, K, R, embed)
        e_exp = e.unsqueeze(2).expand(-1, -1, R, -1)                       # (B, K, R, embed)
        joint = torch.cat([e_exp, i_exp], dim=-1)                          # (B, K, R, 2*embed)
        per_rot = []
        for r in range(R):
            per_rot.append(self.actor_head(joint[:, :, r, :])[:, :, r:r + 1])
        logits = torch.cat(per_rot, dim=-1)                                # (B, K, R)
        logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        flat_logits = logits.reshape(B, K * R)

        # Value: mean-pool over real EMS slots
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(e.dtype)
        pooled = (e * mask.unsqueeze(-1)).sum(dim=1) / denom
        value = self.critic_head(pooled).squeeze(-1)
        return flat_logits, value
