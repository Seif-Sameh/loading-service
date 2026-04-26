"""Smoke tests for the PPO + Transformer stack.

Skipped automatically if torch is not installed (so the slim API image still passes CI).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_packing_transformer_forward_shapes():
    from app.algorithms.rl.packing_transformer import PackingTransformer, PackingTransformerConfig

    m = PackingTransformer(PackingTransformerConfig(embed_dim=32, n_encoder_blocks=1))
    ems = torch.randn(2, 80, 6)
    item = torch.randn(2, 2, 3)
    mask = torch.ones(2, 80, dtype=torch.bool)
    mask[:, 70:] = False
    logits, value = m(ems, item, mask)
    assert logits.shape == (2, 160)
    assert value.shape == (2,)
    # Padding logits must be -inf
    assert torch.isinf(logits[:, 70 * 2 :]).all()


def test_ppo_trainer_one_iter(container_40hc, eur_pallets_10):
    from app.algorithms.rl.packing_transformer import PackingTransformer, PackingTransformerConfig
    from app.algorithms.rl.ppo_trainer import PPOConfig, PPOTrainer

    m = PackingTransformer(PackingTransformerConfig(embed_dim=32, n_encoder_blocks=1))

    def voyage():
        return container_40hc, eur_pallets_10

    trainer = PPOTrainer(
        m,
        sample_voyage_fn=voyage,
        cfg=PPOConfig(n_envs=2, rollout_steps=8, n_epochs=1, minibatch_size=4, log_every=1),
    )
    logs: list[dict] = []
    trainer.train(total_steps=16, on_log=logs.append)
    assert logs, "expected at least one log line"
    assert logs[0]["iter"] >= 1


def test_ppo_save_load_roundtrip(tmp_path, container_40hc, eur_pallets_10):
    from app.algorithms.rl.packing_transformer import PackingTransformer, PackingTransformerConfig
    from app.algorithms.rl.ppo_trainer import PPOConfig, PPOTrainer

    m = PackingTransformer(PackingTransformerConfig(embed_dim=32, n_encoder_blocks=1))
    trainer = PPOTrainer(
        m,
        sample_voyage_fn=lambda: (container_40hc, eur_pallets_10),
        cfg=PPOConfig(n_envs=1, rollout_steps=4, n_epochs=1, minibatch_size=2),
    )
    ckpt = tmp_path / "tiny.pt"
    trainer.save(ckpt)
    loaded = PPOTrainer.load_model(str(ckpt))
    assert isinstance(loaded, PackingTransformer)


def test_ppo_agent_select_runs(tmp_path, container_40hc, eur_pallets_10):
    from app.algorithms.rl.packing_transformer import PackingTransformer, PackingTransformerConfig
    from app.algorithms.rl.ppo_trainer import PPOConfig, PPOTrainer
    from app.algorithms.rl.ppo_agent import PPOPackingAgent
    from app.algorithms.base import solve

    m = PackingTransformer(PackingTransformerConfig(embed_dim=32, n_encoder_blocks=1))
    trainer = PPOTrainer(m, sample_voyage_fn=lambda: (container_40hc, eur_pallets_10),
                        cfg=PPOConfig(n_envs=1, rollout_steps=2, n_epochs=1, minibatch_size=1))
    ckpt = tmp_path / "tiny.pt"
    trainer.save(ckpt)
    agent = PPOPackingAgent(weights_path=ckpt)
    result, _ = solve(algorithm=agent, container=container_40hc, items=eur_pallets_10)
    # Untrained policy may not place all 10, but should make progress.
    assert len(result.placements) >= 1
