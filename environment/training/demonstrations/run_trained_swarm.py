"""Visual inference demo: trained MAPPO swarm in the existing MuJoCo viewer.

  python environment/training/demonstrations/run_trained_swarm.py --checkpoint checkpoints/d2_seed0/final.pt --drones 2
  python environment/training/demonstrations/run_trained_swarm.py --checkpoint checkpoints/stage_d50/final.pt --drones 50 --loop
  python environment/training/demonstrations/run_trained_swarm.py --policy rule --drones 10
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "environment"), str(ROOT / "environment" / "training")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import mujoco
import mujoco.viewer

from rl.world import StaticWorld, Arena
from rl.swarm_env import SwarmEnv, EnvConfig
from rl.observations import OBS_DIM, GLOBAL_STATE_DIM
from evaluate import random_policy, rule_policy


def inject_mocap_drones(xml: str, n: int) -> str:
    """Clone the existing D03 mocap body for drones 4..n (single compile)."""
    if n <= 3:
        return xml
    start = xml.find('<body name="D03" mocap="true"')
    end = xml.find("</body>", start) + len("</body>")
    block = xml[start:end]
    extra = []
    for i in range(4, n + 1):
        b = block.replace("D03", f"D{i:02d}")
        b = "\n".join(l for l in b.split("\n") if "comm_sphere" not in l)
        extra.append(b)
    return xml[:end] + "\n" + "\n".join(extra) + xml[end:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="")
    p.add_argument("--policy", choices=["mappo", "rule", "random"], default="mappo")
    p.add_argument("--drones", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--env-seed", type=int, default=12345)
    p.add_argument("--arena-size", type=float, default=800.0)
    p.add_argument("--mission-seconds", type=float, default=240.0)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--loop", action="store_true", help="auto-restart episodes")
    args = p.parse_args()

    world = StaticWorld(seed=args.env_seed, arena=Arena(size=args.arena_size))
    env = SwarmEnv(world, EnvConfig(num_drones=args.drones, dt=args.dt,
                                    mission_seconds=args.mission_seconds))
    agent = None
    if args.policy == "mappo":
        if not args.checkpoint:
            raise SystemExit("--policy mappo requires --checkpoint")
        import torch
        from mappo import MAPPO
        torch.set_num_threads(1)
        agent = MAPPO(OBS_DIM, GLOBAL_STATE_DIM)
        extra = agent.load(args.checkpoint)
        print(f"[demo] loaded {args.checkpoint} (trained with {extra.get('drones', '?')} drones)")

    def policy(obs):
        if agent is not None:
            state = np.repeat(env.global_state()[None, :], env.num_drones, axis=0)
            a, _, _ = agent.act(obs, state, deterministic=True)
            return np.clip(a, -1, 1)
        rng = np.random.default_rng()
        if args.policy == "rule":
            return rule_policy(obs, env, rng)
        return random_policy(obs, env, rng)

    # One model compile; identical scene geometry to training.
    model = mujoco.MjModel.from_xml_string(inject_mocap_drones(world.xml, args.drones))
    data = mujoco.MjData(model)
    assert model.nmocap >= args.drones

    def geom_id(name):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return gid if gid >= 0 else None

    human_geoms = {}
    for h in world.human_templates:
        ids = [geom_id(f"{h.id}_torso"), geom_id(f"{h.id}_head")]
        human_geoms[h.id] = [g for g in ids if g is not None]
    recolored = set()

    obs, _ = env.reset(seed=args.seed)
    episode = 0
    acc, last, last_hud = 0.0, time.time(), 0.0
    print("[demo] ESC closes the viewer. HUD prints mission progress every 0.5 s.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [world.arena.center_x, world.arena.center_y, 20.0]
        viewer.cam.distance = max(120.0, 30.0 * np.sqrt(args.drones))
        viewer.cam.elevation = -35.0
        viewer.cam.azimuth = 90.0
        while viewer.is_running():
            now = time.time()
            acc += min(now - last, 0.05)
            last = now
            if acc >= env.cfg.dt:
                acc = 0.0
                obs, _, term, trunc, info = env.step(policy(obs))
                if term or trunc:
                    m = info["metrics"]
                    print(f"\n[episode {episode}] {'ALL CONFIRMED' if term else 'time limit'} | "
                          f"discovered {m['discovered']}/{m['discover_targets']} "
                          f"confirmed {m['confirmed']}/{m['confirm_targets']} "
                          f"coverage {m['coverage']:.1%} collisions {m['collision_events']} "
                          f"reward {m['episode_reward_mean']:.1f}")
                    if not args.loop:
                        break
                    episode += 1
                    obs, _ = env.reset(seed=args.seed + episode)
                    recolored.clear()

            # Same mocap sync + kinematics pattern as the existing main.py loop.
            for d in env.drones:
                data.mocap_pos[d.mocap_id] = d.pos
            for vid in env.mission.team_confirmed - recolored:
                for g in human_geoms.get(vid, []):
                    model.geom_rgba[g] = [0.1, 0.95, 0.25, 1.0]
                recolored.add(vid)
            mujoco.mj_kinematics(model, data)
            viewer.sync()

            if now - last_hud >= 0.5:
                last_hud = now
                m = env.metrics()
                warns = [f"{d.id}:{env._sensing[i]['collision_name']}"
                         for i, d in enumerate(env.drones) if env._sensing[i]["colliding"]]
                prox = [d.id for i, d in enumerate(env.drones) if env._sensing[i]["prox_alert"]]
                print(f"t={m['mission_time_s']:6.1f}s | disc {m['discovered']:2d}/{m['discover_targets']} "
                      f"conf {m['confirmed']:2d}/{m['confirm_targets']} | cov {m['coverage']:6.1%} | "
                      f"coll {m['collision_events']:3d} | comms-shares {m['comm_share_events']:3d} | "
                      f"COLLISION: {warns if warns else 'no'} | SEP<10m: {prox if prox else 'ok'}")
            elapsed = time.time() - now
            if elapsed < 0.0166:
                time.sleep(0.0166 - elapsed)
    print("[demo] closed.")


if __name__ == "__main__":
    main()
