"""BC + DAgger warm start from the rule expert.
  python training/pretrain_bc.py --drones 5 --episodes 10 --rounds 3 --out checkpoints/bc_d5_da.pt
"""
import argparse
from pathlib import Path
from config import add_repo_paths, ROOT, MAPPOConfig
add_repo_paths()
import numpy as np
import torch
from rl.world import StaticWorld, Arena
from rl.swarm_env import SwarmEnv, EnvConfig
from rl.observations import OBS_DIM, GLOBAL_STATE_DIM
from evaluate import rule_policy
from mappo import MAPPO
from buffer import RunningMeanStd

p = argparse.ArgumentParser()
p.add_argument("--drones", type=int, default=5)
p.add_argument("--episodes", type=int, default=10, help="episodes per round")
p.add_argument("--rounds", type=int, default=3, help="round 0 = teacher flies; later = clone flies")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--arena-size", type=float, default=800.0)
p.add_argument("--seed", type=int, default=500)
p.add_argument("--out", default="")
a = p.parse_args()

torch.set_num_threads(1); torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
world = StaticWorld(arena=Arena(size=a.arena_size))
env = SwarmEnv(world, EnvConfig(num_drones=a.drones))
agent = MAPPO(OBS_DIM, GLOBAL_STATE_DIM, cfg=MAPPOConfig())
X, Y, S = [], [], []


def clone_action(obs):
    st = np.repeat(env.global_state()[None, :], env.num_drones, axis=0)
    out, _, _ = agent.act(obs, st, deterministic=False)
    return np.clip(out, -1, 1)


def collect(round_idx):
    for ep in range(a.episodes):
        seed = a.seed + 100 * round_idx + ep
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            label = rule_policy(obs, env, rng)              # expert says what to do
            execute = label if round_idx == 0 else clone_action(obs)  # who actually flies
            X.append(obs.copy()); Y.append(label.copy()); S.append(env.global_state().copy())
            obs, _, term, trunc, _ = env.step(execute)
            done = term or trunc
    print(f"round {round_idx}: dataset now {len(X)} env steps "
          f"({len(X) * a.drones} samples), {'teacher' if round_idx == 0 else 'clone'} flying")


def fit():
    Xa = np.concatenate(X); Ya = np.concatenate(Y); Sa = np.stack(S)
    agent.obs_rms = RunningMeanStd((OBS_DIM,)); agent.obs_rms.update(Xa)
    agent.state_rms = RunningMeanStd((GLOBAL_STATE_DIM,)); agent.state_rms.update(Sa)
    Xt = torch.as_tensor(agent.obs_rms.normalize(Xa))
    Yt = torch.as_tensor(Ya, dtype=torch.float32)
    opt = torch.optim.Adam(agent.actor.parameters(), lr=1e-3)
    M = Xa.shape[0]
    for epoch in range(a.epochs):
        perm = rng.permutation(M); tot = 0.0; nb = 0
        for chunk in np.array_split(perm, max(1, M // 2048)):
            mean = torch.tanh(agent.actor.mean(agent.actor.net(Xt[chunk])))
            loss = ((mean - Yt[chunk]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if epoch % 5 == 0 or epoch == a.epochs - 1:
            print(f"  fit epoch {epoch+1:2d}/{a.epochs} | mse {tot/max(nb,1):.4f}")


for r in range(a.rounds):
    collect(r)
    fit()

with torch.no_grad():
    agent.actor.log_std.fill_(-1.0)
out = Path(a.out) if a.out else ROOT / "checkpoints" / f"bc_d{a.drones}_da.pt"
out.parent.mkdir(parents=True, exist_ok=True)
agent.save(out, extra={"dagger_rounds": a.rounds, "samples": len(X) * a.drones,
                       "source": "rule+dagger", "drones": a.drones})
print(f"saved {out}")