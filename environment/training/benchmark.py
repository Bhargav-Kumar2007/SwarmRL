"""Profile actual simulation throughput (no torch, no MuJoCo).

  python environment/training/benchmark.py --drones 2 5 10 25 50 --steps 300
"""
import argparse
import csv
import time

from config import add_repo_paths, ROOT
add_repo_paths()

import numpy as np
from rl.world import StaticWorld, Arena
from rl.swarm_env import SwarmEnv, EnvConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drones", type=int, nargs="+", default=[2, 5, 10, 25, 50])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--env-seed", type=int, default=12345)
    p.add_argument("--arena-size", type=float, default=800.0)
    p.add_argument("--csv", default="")
    args = p.parse_args()
    world = StaticWorld(seed=args.env_seed, arena=Arena(size=args.arena_size))
    rows = []
    for n in args.drones:
        env = SwarmEnv(world, EnvConfig(num_drones=n))
        env.reset(seed=1)
        rng = np.random.default_rng(0)
        env.step(rng.uniform(-1, 1, (n, 3)))          # warmup
        t = time.perf_counter()
        for _ in range(args.steps):
            env.step(rng.uniform(-1, 1, (n, 3)))
        dt = time.perf_counter() - t
        row = {"drones": n, "env_steps_per_s": round(args.steps / dt, 1),
               "agent_steps_per_s": round(args.steps * n / dt, 1),
               "ms_per_step": round(dt / args.steps * 1000, 3)}
        rows.append(row)
        print(row)
    if args.csv:
        path = ROOT / "logs" / args.csv
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
