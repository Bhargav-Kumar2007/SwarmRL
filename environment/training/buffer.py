"""Rollout storage, running feature normalization, GAE, minibatching."""
import numpy as np
import torch


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x):
        x = x.reshape(-1, self.mean.shape[0]).astype(np.float64)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        new_mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot

    def normalize(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, s):
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]


class RolloutBuffer:
    """dones[t] = episode ended AFTER step t (obs[t+1] is then a reset obs)."""

    def __init__(self, T, E, N, obs_dim, state_dim, act_dim, gamma, lam, device):
        self.T, self.E, self.N = T, E, N
        self.gamma, self.lam, self.device = gamma, lam, device
        self.obs = np.zeros((T, E, N, obs_dim), np.float32)
        self.state = np.zeros((T, E, state_dim), np.float32)
        self.actions = np.zeros((T, E, N, act_dim), np.float32)
        self.logp = np.zeros((T, E, N), np.float32)
        self.values = np.zeros((T, E, N), np.float32)
        self.rewards = np.zeros((T, E, N), np.float32)
        self.dones = np.zeros((T, E), bool)
        self.adv = np.zeros((T, E, N), np.float32)
        self.returns = np.zeros((T, E, N), np.float32)

    def add(self, t, obs, state, actions, logp, values, rewards, dones):
        self.obs[t] = obs; self.state[t] = state
        self.actions[t] = actions; self.logp[t] = logp
        self.values[t] = values; self.rewards[t] = rewards; self.dones[t] = dones

    def finalize(self, last_values, last_dones):
        lastgaelam = np.zeros((self.E, self.N), np.float32)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_val, next_nt = last_values, 1.0 - last_dones.astype(np.float32)[:, None]
            else:
                next_val, next_nt = self.values[t + 1], 1.0 - self.dones[t].astype(np.float32)[:, None]
            delta = self.rewards[t] + self.gamma * next_val * next_nt - self.values[t]
            lastgaelam = delta + self.gamma * self.lam * next_nt * lastgaelam
            self.adv[t] = lastgaelam
        self.returns = self.adv + self.values

    def flatten(self, obs_rms, state_rms):
        M = self.T * self.E * self.N
        obs = obs_rms.normalize(self.obs.reshape(M, -1))
        state = state_rms.normalize(np.repeat(self.state, self.N, axis=1).reshape(M, -1))
        to = lambda x, shape=None: torch.as_tensor(
            x.reshape(shape) if shape else x, dtype=torch.float32, device=self.device)
        return {
            "obs": to(obs), "state": to(state),
            "actions": to(self.actions, (M, -1)),
            "old_logp": to(self.logp, (M,)),
            "old_values": to(self.values, (M,)),
            "adv": to(self.adv, (M,)),
            "returns": to(self.returns, (M,)),
        }

    @staticmethod
    def minibatches(data, num_minibatches, rng):
        M = data["obs"].shape[0]
        idx = rng.permutation(M)
        for chunk in np.array_split(idx, num_minibatches):
            yield {k: v[chunk] for k, v in data.items()}