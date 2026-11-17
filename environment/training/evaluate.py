"""Baselines vs trained MAPPO. Exports per-episode CSV, comparison CSV + graphs.

  python training/evaluate.py --compare --drones 2 --episodes 10 \
      --checkpoint checkpoints/d2_seed0/final.pt
"""
import argparse
import csv
import time
from pathlib import Path

from config import add_repo_paths, ROOT
add_repo_paths()

import numpy as np

from rl.world import StaticWorld, Arena
from rl.swarm_env import SwarmEnv, EnvConfig
from rl.observations import OBS_DIM, GLOBAL_STATE_DIM

METRIC_KEYS = ["discovered", "confirmed", "collision_events", "coverage",
               "mission_time_s", "redundant_fraction", "episode_reward_mean"]


def random_policy(obs, env, rng, agent=None):
    return rng.uniform(-1, 1, (env.num_drones, 3))


def rule_policy(obs, env, rng, agent=None):
    """Sensor-driven frontier explorer. Uses only observations, the team
    explored-cell map and obstacle geometry -- never hidden victim positions."""
    acts = np.zeros((env.num_drones, 3))
    a = env.world.arena
    for i, d in enumerate(env.drones):
        o = obs[i]
        if o[16] < 1.0:                       # faint contact in sensor range
            acts[i] = o[17:20]                # 3-D bearing (descends to confirm)
        else:
            best_score, best_dir = -1e9, np.array([1.0, 0.0, 0.0])
            heading = o[3:6]
            base_ang = np.arctan2(heading[1], heading[0]) if np.linalg.norm(heading[:2]) > 0.1 else rng.uniform(-np.pi, np.pi)
            for k in range(8):
                ang = base_ang + k * np.pi / 4.0
                dir3 = np.array([np.cos(ang), np.sin(ang), 0.0])
                probe = d.pos + dir3 * 60.0
                probe[0] = np.clip(probe[0], a.x_bounds[0], a.x_bounds[1])
                probe[1] = np.clip(probe[1], a.y_bounds[0], a.y_bounds[1])
                cell = env.mission.cell_of(probe)
                score = 1.0 - float(env.mission.team_visited[cell])
                mid = d.pos + dir3 * 30.0
                near = np.linalg.norm(env.matrix - mid, axis=1) - env.radii
                for j in np.flatnonzero(near < 8.0)[:5]:
                    if env.obstacles[int(j)].surface_distance(mid) < 6.0:
                        score -= 1.0
                if k == 0:
                    score += 0.1              # slight persistence bias
                if score > best_score:
                    best_score, best_dir = score, dir3
            acts[i] = best_dir
        if o[16] >= 1.0 or o[16] * 50.0 > 15.0:
            acts[i, 2] += np.clip((12.0 - d.pos[2]) / 20.0, -0.5, 0.5)
        if o[7] * 100.0 < 10.0:               # obstacle repulsion
            acts[i] -= o[8:11] * 0.8
        for k in range(3):                    # drone separation repulsion
            b = 24 + 4 * k
            if o[b + 3] > 0.5:
                rel = o[b:b + 3] * d.comm_radius
                n = np.linalg.norm(rel)
                if 1e-6 < n < 20.0:
                    acts[i] -= rel / n * 0.5
    norms = np.linalg.norm(acts, axis=1, keepdims=True)
    return acts / np.maximum(norms, 1.0)


def mappo_policy(obs, env, rng, agent):
    state = np.repeat(env.global_state()[None, :], env.num_drones, axis=0)
    a, _, _ = agent.act(obs, state, deterministic=True)
    return np.clip(a, -1, 1)


def run_policy(name, policy, world, args, agent=None):
    env = SwarmEnv(world, EnvConfig(num_drones=args.drones, dt=args.dt,
                                    mission_seconds=args.mission_seconds))
    rng = np.random.default_rng(args.seed)
    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        t0 = time.perf_counter()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs, env, rng, agent))
            done = term or trunc
        wall = time.perf_counter() - t0
        m = info["metrics"]
        row = {"policy": name, "episode": ep, "env_sps": round(m["steps"] / wall, 1)}
        row.update({k: m[k] for k in METRIC_KEYS})
        rows.append(row)
        print(f"  [{name}] ep {ep}: disc {m['discovered']:2d} conf {m['confirmed']:2d} "
              f"cov {m['coverage']:.3f} coll {m['collision_events']:3d} "
              f"rew {m['episode_reward_mean']:8.2f} | {row['env_sps']} env sps")
    return rows


def write_csv(path, rows):
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["random", "rule", "mappo"], default="random")
    p.add_argument("--compare", action="store_true", help="run all policies")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--drones", type=int, default=2)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--env-seed", type=int, default=12345)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--mission-seconds", type=float, default=240.0)
    p.add_argument("--arena-size", type=float, default=800.0)
    p.add_argument("--run-name", default="eval")
    args = p.parse_args()

    world = StaticWorld(seed=args.env_seed, arena=Arena(size=args.arena_size))
    out = ROOT / "logs" / args.run_name
    out.mkdir(parents=True, exist_ok=True)

    agent = None
    if args.policy == "mappo" or args.compare:
        if args.checkpoint:
            import torch
            from mappo import MAPPO
            torch.set_num_threads(1)
            agent = MAPPO(OBS_DIM, GLOBAL_STATE_DIM)
            agent.load(args.checkpoint)

    policies = [("random", random_policy), ("rule", rule_policy)]
    if agent is not None:
        policies.append(("mappo", mappo_policy))
    if not args.compare:
        policies = [(args.policy, {"random": random_policy, "rule": rule_policy,
                                   "mappo": mappo_policy}[args.policy])]
        if args.policy == "mappo" and agent is None:
            raise SystemExit("--policy mappo requires --checkpoint")

    all_rows, summary = [], []
    for name, pol in policies:
        rows = run_policy(name, pol, world, args, agent)
        write_csv(out / f"eval_{name}.csv", rows)
        all_rows.extend(rows)
        summary.append({"policy": name,
                        **{k: round(float(np.mean([r[k] for r in rows])), 4) for k in METRIC_KEYS},
                        "env_sps": round(float(np.mean([r["env_sps"] for r in rows])), 1)})
    write_csv(out / "comparison.csv", summary)
    print("\n=== comparison (mean over episodes) ===")
    for s in summary:
        print(s)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = ["discovered", "confirmed", "coverage", "collision_events",
                "redundant_fraction", "episode_reward_mean"]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        names = [s["policy"] for s in summary]
        for ax, k in zip(axes.flat, keys):
            ax.bar(names, [s[k] for s in summary])
            ax.set_title(k)
        fig.tight_layout()
        fig.savefig(out / "comparison.png", dpi=140)
        print(f"graphs -> {out / 'comparison.png'}")
    except ImportError:
        print("matplotlib not installed -- CSV only (pip install matplotlib for graphs)")


if __name__ == "__main__":
    main()