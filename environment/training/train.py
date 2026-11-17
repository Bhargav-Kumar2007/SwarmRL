"""Headless MAPPO training with optional curriculum (2 -> 5 -> 10 -> 25 -> 50).

Run from the repo root, e.g.:
  python training/train.py --drones 2 --num-envs 4 --total-steps 300000
  python training/train.py --curriculum --num-envs 4
"""
import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from config import add_repo_paths, MAPPOConfig, CURRICULUM_STAGES, DEFAULT_STAGE_STEPS, ROOT
add_repo_paths()

import numpy as np
import torch

from rl.world import StaticWorld, Arena
from rl.swarm_env import EnvConfig, VecSwarmEnv
from rl.observations import OBS_DIM, GLOBAL_STATE_DIM
from rl.rewards import RewardWeights
from mappo import MAPPO
from buffer import RolloutBuffer


def parse_args():
    p = argparse.ArgumentParser(description="SwarmRL MAPPO trainer")
    p.add_argument("--drones", type=int, default=2)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--total-steps", type=int, default=300_000, help="agent transitions")
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-seed", type=int, default=12345, help="procedural city seed")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--resume", default="")
    p.add_argument("--run-name", default="")
    p.add_argument("--checkpoint-interval", type=int, default=10, help="iterations")
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--arena-size", type=float, default=800.0)
    p.add_argument("--arena-x", type=float, default=260.0)
    p.add_argument("--arena-y", type=float, default=160.0)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--mission-seconds", type=float, default=240.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    return p.parse_args()


def run_stage(args, world, n_drones, budget, resume_path, run_name):
    run_dir = ROOT / "logs" / run_name
    ck_dir = ROOT / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = EnvConfig(num_drones=n_drones, dt=args.dt, mission_seconds=args.mission_seconds)
    mcfg = MAPPOConfig(lr=args.lr, entropy_coef=args.entropy_coef)
    (run_dir / "config.json").write_text(json.dumps(
        {"args": vars(args), "env": {k: v for k, v in asdict(env_cfg).items() if k != "reward_weights"},
         "reward_weights": asdict(env_cfg.reward_weights), "mappo": asdict(mcfg)}, indent=2))

    vec = VecSwarmEnv(world, env_cfg, args.num_envs, base_seed=args.seed)
    agent = MAPPO(OBS_DIM, GLOBAL_STATE_DIM, cfg=mcfg, device=args.device)
    if resume_path:
        extra = agent.load(resume_path)
        print(f"[stage {n_drones}] transferred checkpoint {resume_path} (extra={extra})")

    E, N, T = args.num_envs, n_drones, args.rollout_steps
    buf = RolloutBuffer(T, E, N, OBS_DIM, GLOBAL_STATE_DIM, 3,
                        mcfg.gamma, mcfg.gae_lambda, args.device)
    obs, states, _ = vec.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    csv_path = run_dir / "train.csv"
    fields = ["iter", "agent_steps", "env_sps", "agent_sps", "episodes",
              "ep_reward", "discovered", "confirmed", "coverage", "collisions",
              "redundant_frac", "policy_loss", "value_loss", "entropy", "approx_kl"]
    fh = csv_path.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()

    agent_steps, it, recent = 0, 0, []
    while agent_steps < budget:
        t_start = time.perf_counter()
        for t in range(T):
            a, logp, v = agent.act(obs.reshape(-1, OBS_DIM),
                                   np.repeat(states, N, axis=0))
            next_obs, rew, dones, next_states, _, finals = vec.step(
                np.clip(a, -1, 1).reshape(E, N, 3))
            buf.add(t, obs, states, a.reshape(E, N, 3), logp.reshape(E, N),
                    v.reshape(E, N), rew, dones)
            obs, states = next_obs, next_states
            recent.extend(finals)
        collect_s = time.perf_counter() - t_start
        _, _, last_v = agent.act(obs.reshape(-1, OBS_DIM), np.repeat(states, N, axis=0))
        buf.finalize(last_v.reshape(E, N), dones)
        stats = agent.update(buf, np.random.default_rng(args.seed + 10_000 + it))
        agent_steps += T * E * N
        it += 1

        def ep_mean(key):
            return float(np.mean([f[key] for f in recent])) if recent else 0.0

        row = {"iter": it, "agent_steps": agent_steps,
               "env_sps": round(T * E / collect_s, 1),
               "agent_sps": round(T * E * N / collect_s, 1),
               "episodes": len(recent),
               "ep_reward": round(ep_mean("episode_reward_mean"), 3),
               "discovered": round(ep_mean("discovered"), 2),
               "confirmed": round(ep_mean("confirmed"), 2),
               "coverage": round(ep_mean("coverage"), 4),
               "collisions": round(ep_mean("collision_events"), 2),
               "redundant_frac": round(ep_mean("redundant_fraction"), 4),
               "policy_loss": round(stats["policy_loss"], 5),
               "value_loss": round(stats["value_loss"], 5),
               "entropy": round(stats["entropy"], 4),
               "approx_kl": round(stats["approx_kl"], 5)}
        writer.writerow(row)
        fh.flush()
        recent = []
        print(f"[{run_name}] it {it:4d} | steps {agent_steps:8d} | "
              f"env {row['env_sps']:7.1f} sps | ep_rew {row['ep_reward']:8.2f} | "
              f"disc {row['discovered']:5.2f} conf {row['confirmed']:5.2f} "
              f"cov {row['coverage']:.3f} coll {row['collisions']:5.2f}")
        if it % args.checkpoint_interval == 0:
            agent.save(ck_dir / "latest.pt",
                       extra={"agent_steps": agent_steps, "drones": n_drones, "iter": it})
    fh.close()
    final = ck_dir / "final.pt"
    agent.save(final, extra={"agent_steps": agent_steps, "drones": n_drones, "iter": it})
    print(f"[{run_name}] saved {final}")
    return str(final)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    world = StaticWorld(seed=args.env_seed,
                        arena=Arena(center_x=args.arena_x, center_y=args.arena_y,
                                    size=args.arena_size))
    print(f"[world] {len(world.obstacles)} obstacles | arena {args.arena_size:.0f} m | "
          f"{len(world.discover_target_ids)} discover / {len(world.confirm_target_ids)} confirm targets")
    if args.curriculum:
        prev = args.resume or None
        for n in CURRICULUM_STAGES:
            prev = run_stage(args, world, n, DEFAULT_STAGE_STEPS[n], prev, f"stage_d{n}")
            print(f"[curriculum] stage N={n} done. Evaluate before trusting the next stage:")
            print(f"  python training/evaluate.py --compare --drones {n} "
                  f"--checkpoint checkpoints/stage_d{n}/final.pt --episodes 10")
    else:
        run_name = args.run_name or f"d{args.drones}_seed{args.seed}"
        run_stage(args, world, args.drones, args.total_steps, args.resume or None, run_name)


if __name__ == "__main__":
    main()