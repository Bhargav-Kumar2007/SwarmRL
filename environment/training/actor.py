"""One shared actor for every drone (parameter sharing)."""
import torch
import torch.nn as nn
from torch.distributions import Normal


def init_layer(layer, gain=2 ** 0.5):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, act_dim)
        for m in self.net:
            if isinstance(m, nn.Linear):
                init_layer(m)
        init_layer(self.mean, gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs):
        mean = torch.tanh(self.mean(self.net(obs)))
        std = torch.exp(self.log_std.clamp(-2.5, 0.5)).expand_as(mean)
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1)

    def evaluate(self, obs, actions):
        d = self.dist(obs)
        return d.log_prob(actions).sum(-1), d.entropy().sum(-1)