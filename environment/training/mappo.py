"""MAPPO: shared actor + centralized critic, clipped PPO updates."""
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn

from config import MAPPOConfig
from actor import Actor
from critic import Critic
from buffer import RunningMeanStd


class MAPPO:
    def __init__(self, obs_dim, state_dim, act_dim=3, cfg=None, device="cpu"):
        self.cfg = cfg or MAPPOConfig()
        self.obs_dim, self.state_dim, self.act_dim = obs_dim, state_dim, act_dim
        self.device = torch.device(device)
        self.actor = Actor(obs_dim, act_dim, self.cfg.actor_hidden).to(self.device)
        self.critic = Critic(obs_dim + state_dim, self.cfg.critic_hidden).to(self.device)
        self.params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.opt = torch.optim.Adam(self.params, lr=self.cfg.lr, eps=1e-5)
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.state_rms = RunningMeanStd((state_dim,))

    @torch.no_grad()
    def act(self, obs_np, state_np, deterministic=False):
        ot = torch.as_tensor(self.obs_rms.normalize(obs_np), device=self.device)
        st = torch.as_tensor(self.state_rms.normalize(state_np), device=self.device)
        a, logp = self.actor.act(ot, deterministic)
        v = self.critic(torch.cat([st, ot], dim=-1))
        return a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()

    def update(self, buffer, rng):
        self.obs_rms.update(buffer.obs)
        self.state_rms.update(buffer.state)
        data = buffer.flatten(self.obs_rms, self.state_rms)
        c = self.cfg
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0}
        n_updates = 0
        for _ in range(c.ppo_epochs):
            for mb in buffer.minibatches(data, c.num_minibatches, rng):
                logp, ent = self.actor.evaluate(mb["obs"], mb["actions"])
                v = self.critic(torch.cat([mb["state"], mb["obs"]], dim=-1))
                ratio = torch.exp(logp - mb["old_logp"])
                adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
                pg_loss = torch.max(-adv * ratio,
                                    -adv * ratio.clamp(1 - c.clip_param, 1 + c.clip_param)).mean()
                v_clipped = mb["old_values"] + (v - mb["old_values"]).clamp(-c.value_clip, c.value_clip)
                v_loss = 0.5 * torch.max((v - mb["returns"]) ** 2,
                                         (v_clipped - mb["returns"]) ** 2).mean()
                loss = pg_loss + c.value_coef * v_loss - c.entropy_coef * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, c.max_grad_norm)
                self.opt.step()
                stats["policy_loss"] += pg_loss.item()
                stats["value_loss"] += v_loss.item()
                stats["entropy"] += ent.mean().item()
                stats["approx_kl"] += (mb["old_logp"] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1).abs() > c.clip_param).float().mean().item()
                n_updates += 1
        return {k: v / max(n_updates, 1) for k, v in stats.items()}

    def save(self, path, extra=None):
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "obs_rms": self.obs_rms.state_dict(), "state_rms": self.state_rms.state_dict(),
            "obs_dim": self.obs_dim, "state_dim": self.state_dim, "act_dim": self.act_dim,
            "cfg": asdict(self.cfg), "extra": extra or {},
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        assert ck["obs_dim"] == self.obs_dim and ck["state_dim"] == self.state_dim, \
            "observation dims changed -- checkpoint incompatible"
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.obs_rms.load_state_dict(ck["obs_rms"])
        self.state_rms.load_state_dict(ck["state_rms"])
        return ck.get("extra", {})