"""One centralized critic: input = global state + this drone's observation."""
import torch.nn as nn
from actor import init_layer


class Critic(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 64), nn.Tanh(),
                                 nn.Linear(64, 1))
        init_layer(self.net[0]); init_layer(self.net[2])
        init_layer(self.net[4], gain=1.0)

    def forward(self, x):
        return self.net(x).squeeze(-1)