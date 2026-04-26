"""PPO trainer for :class:`PackingTransformer`.

Self-contained — no stable-baselines3 dependency so the student can defend every line in
the thesis. Implements:

- on-policy rollouts of full episodes (one container per episode)
- Generalized Advantage Estimation (GAE-λ)
- clipped PPO surrogate objective with entropy bonus
- value-function loss with optional clipping
- vectorisation across `n_envs` parallel containers (CPU only by default; flip ``device``
  to use GPU)

Usage from the training notebook:

```python
trainer = PPOTrainer(model, sample_voyage_fn=...)
trainer.train(total_steps=1_000_000)
trainer.save("models/gopt_v1.pt")
```
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from app.algorithms.rl.packing_transformer import PackingTransformer
from app.catalog.loader import get_container
from app.env.packing_env import PackingEnv
from app.schemas import CargoItem, Container


SampleVoyageFn = Callable[[], tuple[Container, list[CargoItem]]]


@dataclass
class PPOConfig:
    n_envs: int = 8
    rollout_steps: int = 128       # max steps per env per rollout
    n_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "cpu"
    log_every: int = 10            # rollout iterations between log prints


@dataclass
class _RolloutBuffer:
    obs_ems: list = field(default_factory=list)
    obs_item: list = field(default_factory=list)
    obs_mask: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)


class PPOTrainer:
    """Train a :class:`PackingTransformer` with PPO.

    Parameters
    ----------
    model:
        Policy + value network. Must accept the env's observation tensors.
    sample_voyage_fn:
        Zero-arg callable that returns a fresh (container, items) pair each episode. The
        :class:`AlexandriaSampler` in :mod:`app.data.alexandria_sampler` is the production
        choice; for unit tests pass a deterministic fixture.
    cfg:
        Training hyper-parameters.
    """

    def __init__(
        self,
        model: PackingTransformer,
        sample_voyage_fn: SampleVoyageFn,
        cfg: PPOConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or PPOConfig()
        self.sample_voyage_fn = sample_voyage_fn
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self._envs: list[PackingEnv] = []
        self._reset_envs()

    # ----- env management -----

    def _reset_envs(self) -> list[dict[str, np.ndarray]]:
        self._envs = []
        first_obs: list[dict[str, np.ndarray]] = []
        for _ in range(self.cfg.n_envs):
            cont, items = self.sample_voyage_fn()
            env = PackingEnv(container=cont, items=items, max_candidates=80)
            obs, _ = env.reset()
            self._envs.append(env)
            first_obs.append(obs)
        return first_obs

    def _restart_env(self, idx: int) -> dict[str, np.ndarray]:
        cont, items = self.sample_voyage_fn()
        env = PackingEnv(container=cont, items=items, max_candidates=80)
        obs, _ = env.reset()
        self._envs[idx] = env
        return obs

    # ----- inference helpers -----

    @staticmethod
    def _stack_obs(obs_list: list[dict[str, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ems = torch.from_numpy(np.stack([o["ems"] for o in obs_list])).float()
        item = torch.from_numpy(np.stack([o["item"] for o in obs_list])).float()
        mask = torch.from_numpy(np.stack([o["mask"] for o in obs_list])).bool()
        return ems, item, mask

    def _act(
        self, obs_list: list[dict[str, np.ndarray]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ems, item, mask = self._stack_obs(obs_list)
        ems, item, mask = ems.to(self.device), item.to(self.device), mask.to(self.device)
        logits, value = self.model(ems, item, mask)
        # Build a 2K mask: each EMS slot has n_rotations actions
        n_rot = self.model.cfg.n_rotations
        full_mask = mask.unsqueeze(-1).expand(-1, -1, n_rot).reshape(mask.size(0), -1)
        logits = logits.masked_fill(~full_mask, float("-inf"))
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, full_mask

    # ----- main rollout -----

    def collect_rollout(self) -> tuple[_RolloutBuffer, list[float], list[float]]:
        buf = _RolloutBuffer()
        ep_returns: list[float] = []
        ep_utilizations: list[float] = []
        running_rewards = [0.0] * self.cfg.n_envs

        obs_list: list[dict[str, np.ndarray]] = [env._obs() for env in self._envs]
        for _ in range(self.cfg.rollout_steps):
            buf.obs_ems.append(np.stack([o["ems"] for o in obs_list]))
            buf.obs_item.append(np.stack([o["item"] for o in obs_list]))
            buf.obs_mask.append(np.stack([o["mask"] for o in obs_list]))

            action, log_prob, value, _ = self._act(obs_list)
            buf.actions.append(action.detach().cpu().numpy())
            buf.log_probs.append(log_prob.detach().cpu().numpy())
            buf.values.append(value.detach().cpu().numpy())

            n_rot = self.model.cfg.n_rotations
            step_rewards = np.zeros(self.cfg.n_envs, dtype=np.float32)
            step_dones = np.zeros(self.cfg.n_envs, dtype=np.float32)
            for i, env in enumerate(self._envs):
                a = int(action[i].item())
                cand_idx, _rot = divmod(a, n_rot)  # rotation embedded in action; env picks via cand_idx for now
                if cand_idx >= len(env.state.candidates):
                    cand_idx = 0  # action mask should prevent this; fallback
                obs, r, done, _, _ = env.step(cand_idx)
                step_rewards[i] = float(r)
                step_dones[i] = 1.0 if done else 0.0
                running_rewards[i] += float(r)
                if done:
                    ep_returns.append(running_rewards[i])
                    score, kpis = env.final_score()
                    ep_utilizations.append(kpis.utilization)
                    running_rewards[i] = 0.0
                    obs = self._restart_env(i)
                obs_list[i] = obs
            buf.rewards.append(step_rewards)
            buf.dones.append(step_dones)

        # Bootstrap value for the last observation
        with torch.no_grad():
            _, _, last_value, _ = self._act(obs_list)
        buf.values.append(last_value.detach().cpu().numpy())
        return buf, ep_returns, ep_utilizations

    # ----- learning step -----

    def _compute_advantages(self, buf: _RolloutBuffer) -> tuple[np.ndarray, np.ndarray]:
        T = self.cfg.rollout_steps
        rewards = np.asarray(buf.rewards, dtype=np.float32)
        values = np.asarray(buf.values, dtype=np.float32)
        dones = np.asarray(buf.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = np.zeros(self.cfg.n_envs, dtype=np.float32)
        for t in reversed(range(T)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * values[t + 1] * non_terminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self, buf: _RolloutBuffer) -> dict[str, float]:
        T = self.cfg.rollout_steps
        N = self.cfg.n_envs
        advantages, returns = self._compute_advantages(buf)
        flat = lambda arr: np.asarray(arr).reshape(T * N, *np.asarray(arr).shape[2:])

        ems_b = torch.from_numpy(flat(buf.obs_ems)).float().to(self.device)
        item_b = torch.from_numpy(flat(buf.obs_item)).float().to(self.device)
        mask_b = torch.from_numpy(flat(buf.obs_mask)).bool().to(self.device)
        actions_b = torch.from_numpy(flat(buf.actions)).long().to(self.device)
        old_logp_b = torch.from_numpy(flat(buf.log_probs)).float().to(self.device)
        old_val_b = torch.from_numpy(flat(buf.values[:-1])).float().to(self.device)
        adv_b = torch.from_numpy(flat(advantages)).float().to(self.device)
        ret_b = torch.from_numpy(flat(returns)).float().to(self.device)

        # Normalise advantages
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        idx = np.arange(T * N)
        n_minibatch = max(1, (T * N) // self.cfg.minibatch_size)
        loss_log = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
        for _ in range(self.cfg.n_epochs):
            np.random.shuffle(idx)
            for mb_start in range(0, T * N, self.cfg.minibatch_size):
                mb = idx[mb_start:mb_start + self.cfg.minibatch_size]
                mb_t = torch.from_numpy(mb).long().to(self.device)
                logits, value = self.model(ems_b[mb_t], item_b[mb_t], mask_b[mb_t])
                n_rot = self.model.cfg.n_rotations
                full_mask = mask_b[mb_t].unsqueeze(-1).expand(-1, -1, n_rot).reshape(len(mb), -1)
                logits = logits.masked_fill(~full_mask, float("-inf"))
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(actions_b[mb_t])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logp - old_logp_b[mb_t])
                surr1 = ratio * adv_b[mb_t]
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv_b[mb_t]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                v_clipped = old_val_b[mb_t] + torch.clamp(
                    value - old_val_b[mb_t], -self.cfg.value_clip_eps, self.cfg.value_clip_eps
                )
                v_loss_1 = (value - ret_b[mb_t]).pow(2)
                v_loss_2 = (v_clipped - ret_b[mb_t]).pow(2)
                value_loss = 0.5 * torch.max(v_loss_1, v_loss_2).mean()

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                loss_log["policy"] += float(policy_loss.item())
                loss_log["value"] += float(value_loss.item())
                loss_log["entropy"] += float(entropy.item())

        denom = self.cfg.n_epochs * n_minibatch
        return {k: v / denom for k, v in loss_log.items()}

    # ----- public train loop -----

    def train(self, total_steps: int, on_log: Callable[[dict], None] | None = None) -> None:
        steps_done = 0
        rollout_iter = 0
        while steps_done < total_steps:
            buf, ep_returns, ep_utils = self.collect_rollout()
            losses = self.update(buf)
            steps_done += self.cfg.rollout_steps * self.cfg.n_envs
            rollout_iter += 1
            if on_log and (rollout_iter % self.cfg.log_every == 0 or rollout_iter == 1):
                on_log({
                    "iter": rollout_iter,
                    "steps_done": steps_done,
                    "episodes": len(ep_returns),
                    "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
                    "mean_util": float(np.mean(ep_utils)) if ep_utils else 0.0,
                    **losses,
                })

    # ----- io -----

    def save(self, path) -> None:
        torch.save(
            {"model_state": self.model.state_dict(), "cfg": vars(self.model.cfg)},
            path,
        )

    @classmethod
    def load_model(cls, path: str, *, device: str = "cpu") -> PackingTransformer:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        from app.algorithms.rl.packing_transformer import PackingTransformerConfig
        cfg = PackingTransformerConfig(**ckpt["cfg"])
        model = PackingTransformer(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model
