"""Visual inference demo: trained swarm in the existing MuJoCo viewer.

  python environment/training/demonstrations/run_trained_swarm.py --checkpoint checkpoints/d5_v2/final.pt --drones 10
  python environment/training/demonstrations/run_trained_swarm.py --policy rule --drones 25 --follow
  python environment/training/demonstrations/run_trained_swarm.py --checkpoint checkpoints/d5_v2/final.pt --drones 50 --loop

Fixes vs old version:
  * removes the scene's stock D01/D02/D03 mocap bodies (no phantom drone underground)
  * builds EXACTLY N mocap bodies, verified against model.nmocap
  * drones scaled up so they are visible at arena scale (visual only)
  * camera frames the whole arena (or follows the swarm with --follow)
"""
import argparse
import re
import sys
import time
from pathlib import Path

# ---- robust repo/path resolution (works for any nesting layout) ----
HERE = Path(__file__).resolve()
ROOT = next((c for c in HERE.parents if (c / "environment" / "rl").is_dir()), None)
if ROOT is None:
    ROOT = HERE.parents[3]
TRAIN_DIR = next((c for c in (ROOT / "training", ROOT / "environment" / "training", HERE.parent)
                  if (c / "evaluate.py").exists()), HERE.parent)
for p in (str(ROOT), str(ROOT / "environment"), str(TRAIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import mujoco
import mujoco.viewer

from rl.world import StaticWorld, Arena
from rl.swarm_env import SwarmEnv, EnvConfig
from rl.observations import OBS_DIM, GLOBAL_STATE_DIM
from evaluate import random_policy, rule_policy

# matches <body name="D01" ... mocap="true" ...> ... </body>  (no nested bodies inside)
STOCK_DRONE_RE = re.compile(r'\s*<body\s+name="D\d+"[^>]*mocap="true"[^>]*>.*?</body>', re.DOTALL)

PALETTE = [(0.2, 0.95, 0.3), (0.95, 0.6, 0.1), (0.85, 0.25, 0.9), (0.25, 0.7, 1.0),
           (1.0, 0.9, 0.2), (1.0, 0.4, 0.6), (0.4, 1.0, 0.85), (0.95, 0.95, 0.95)]


def drone_body_xml(i: int, s: float) -> str:
    """One mocap drone. s = visual scale (collision radius in the env is unchanged)."""
    n = f"SD{i:03d}"
    r, g, b = PALETTE[i % len(PALETTE)]
    a = 1.5 * s
    return f"""
    <body name="{n}" mocap="true" pos="0 0 -500">
      <geom name="{n}_hull" type="cylinder" size="{0.9*s:.2f} {0.25*s:.2f}" material="drone_chassis" contype="0" conaffinity="0"/>
      <geom name="{n}_dome" type="sphere" pos="0 0 {0.45*s:.2f}" size="{0.6*s:.2f}" rgba="{r:.2f} {g:.2f} {b:.2f} 1" contype="0" conaffinity="0"/>
      <geom name="{n}_beacon" type="sphere" pos="0 0 {1.6*s:.2f}" size="{0.32*s:.2f}" rgba="1 1 0.15 1" contype="0" conaffinity="0"/>
      <geom name="{n}_arm1" type="capsule" fromto="{-a:.2f} {-a:.2f} 0 {a:.2f} {a:.2f} 0" size="{0.12*s:.2f}" material="drone_arm" contype="0" conaffinity="0"/>
      <geom name="{n}_arm2" type="capsule" fromto="{-a:.2f} {a:.2f} 0 {a:.2f} {-a:.2f} 0" size="{0.12*s:.2f}" material="drone_arm" contype="0" conaffinity="0"/>
      <geom name="{n}_r1" type="cylinder" pos="{a:.2f} {a:.2f} {0.12*s:.2f}" size="{0.65*s:.2f} {0.04*s:.2f}" material="rotor_front" contype="0" conaffinity="0"/>
      <geom name="{n}_r2" type="cylinder" pos="{a:.2f} {-a:.2f} {0.12*s:.2f}" size="{0.65*s:.2f} {0.04*s:.2f}" material="rotor_front" contype="0" conaffinity="0"/>
      <geom name="{n}_r3" type="cylinder" pos="{-a:.2f} {a:.2f} {0.12*s:.2f}" size="{0.65*s:.2f} {0.04*s:.2f}" material="rotor_rear" contype="0" conaffinity="0"/>
      <geom name="{n}_r4" type="cylinder" pos="{-a:.2f} {-a:.2f} {0.12*s:.2f}" size="{0.65*s:.2f} {0.04*s:.2f}" material="rotor_rear" contype="0" conaffinity="0"/>
    </body>"""


def build_swarm_xml(base_xml: str, n: int, scale: float) -> str:
    """Strip stock drones, append exactly n fresh mocap drones at end of worldbody."""
    xml, removed = STOCK_DRONE_RE.subn("", base_xml)
    idx = xml.rfind("</worldbody>")
    if idx < 0:
        raise RuntimeError("could not locate </worldbody> in generated scene XML")
    bodies = "".join(drone_body_xml(i, scale) for i in range(n))
    print(f"[demo] removed {removed} stock drone bodies; added {n} swarm drones")
    return xml[:idx] + bodies + "\n  " + xml[idx:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="")
    p.add_argument("--policy", choices=["mappo", "rule", "random"], default="")
    p.add_argument("--drones", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--env-seed", type=int, default=12345)
    p.add_argument("--arena-size", type=float, default=800.0)
    p.add_argument("--mission-seconds", type=float, default=240.0)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--drone-scale", type=float, default=0.0,
                   help="visual size multiplier (0 = auto: arena/200). Collision rules unchanged.")
    p.add_argument("--follow", action="store_true", help="camera follows swarm centroid")
    p.add_argument("--cam-distance", type=float, default=0.0, help="0 = auto")
    p.add_argument("--time-scale", type=float, default=1.0, help="run faster than real time")
    p.add_argument("--loop", action="store_true")
    args = p.parse_args()

    policy_name = args.policy or ("mappo" if args.checkpoint else "rule")
    scale = args.drone_scale if args.drone_scale > 0 else max(1.0, args.arena_size / 200.0)

    world = StaticWorld(seed=args.env_seed, arena=Arena(size=args.arena_size))
    env = SwarmEnv(world, EnvConfig(num_drones=args.drones, dt=args.dt,
                                    mission_seconds=args.mission_seconds))

    agent = None
    if policy_name == "mappo":
        if not args.checkpoint:
            raise SystemExit("--policy mappo requires --checkpoint")
        import torch
        from mappo import MAPPO
        torch.set_num_threads(1)
        agent = MAPPO(OBS_DIM, GLOBAL_STATE_DIM)
        extra = agent.load(args.checkpoint)
        print(f"[demo] loaded {args.checkpoint} (trained with {extra.get('drones', '?')} drones)")

    rng = np.random.default_rng(args.seed)

    def policy(obs):
        if agent is not None:
            state = np.repeat(env.global_state()[None, :], env.num_drones, axis=0)
            a, _, _ = agent.act(obs, state, deterministic=True)
            return np.clip(a, -1, 1)
        if policy_name == "rule":
            return rule_policy(obs, env, rng)
        return random_policy(obs, env, rng)

    # ---------- one model compile, exactly N mocap drones ----------
    model = mujoco.MjModel.from_xml_string(build_swarm_xml(world.xml, args.drones, scale))
    data = mujoco.MjData(model)
    print(f"[demo] requested drones = {args.drones} | model.nmocap = {model.nmocap}")
    if model.nmocap != args.drones:
        raise SystemExit(f"mocap count mismatch ({model.nmocap} != {args.drones}) -- XML build failed")

    def geom_id(name):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return gid if gid >= 0 else None

    dome_ids, base_colors = [], []
    for i in range(args.drones):
        gid = geom_id(f"SD{i:03d}_dome")
        if gid is None:
            raise SystemExit(f"missing dome geom for drone {i}")
        dome_ids.append(gid)
        base_colors.append(np.array(model.geom_rgba[gid], dtype=np.float32).copy())
    print(f"[demo] drone geoms resolved: {len(dome_ids)}/{args.drones} | visual scale {scale:.1f}x "
          f"(collision radius still {env.drones[0].collision_radius} m)")

    human_geoms = {}
    for h in world.human_templates:
        ids = [geom_id(f"{h.id}_torso"), geom_id(f"{h.id}_head")]
        human_geoms[h.id] = [g for g in ids if g is not None]
    seen_discovered, seen_confirmed = set(), set()

    obs, _ = env.reset(seed=args.seed)
    print("[demo] initial drone positions:")
    for d in env.drones:
        print(f"   {d.id}  ({d.pos[0]:7.1f}, {d.pos[1]:7.1f}, {d.pos[2]:5.1f})")

    episode = 0
    acc, last, last_hud = 0.0, time.time(), 0.0
    print("[demo] ESC closes the viewer. HUD every 0.5 s. Domes: red=collision, orange=separation alert.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [world.arena.center_x, world.arena.center_y, 20.0]
        viewer.cam.distance = args.cam_distance or (150.0 if args.follow else args.arena_size * 1.2)
        viewer.cam.elevation = -45.0 if not args.follow else -30.0
        viewer.cam.azimuth = 90.0
        while viewer.is_running():
            now = time.time()
            acc += min(now - last, 0.05) * args.time_scale
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
                    seen_discovered.clear(); seen_confirmed.clear()

            for d in env.drones:
                data.mocap_pos[d.mocap_id] = d.pos

            for vid in env.mission.team_discovered - seen_discovered - seen_confirmed:
                for g in human_geoms.get(vid, []):
                    model.geom_rgba[g] = [1.0, 0.85, 0.1, 1.0]     # discovered = yellow
                seen_discovered.add(vid)
            for vid in env.mission.team_confirmed - seen_confirmed:
                for g in human_geoms.get(vid, []):
                    model.geom_rgba[g] = [0.1, 0.95, 0.25, 1.0]    # confirmed = green
                seen_confirmed.add(vid)

            for i, gid in enumerate(dome_ids):
                sen = env._sensing[i]
                if sen["colliding"]:
                    model.geom_rgba[gid] = [1.0, 0.1, 0.1, 1.0]
                elif sen["prox_alert"]:
                    model.geom_rgba[gid] = [1.0, 0.55, 0.0, 1.0]
                else:
                    model.geom_rgba[gid] = base_colors[i]

            if args.follow:
                viewer.cam.lookat[:] = np.mean([d.pos for d in env.drones], axis=0)

            mujoco.mj_kinematics(model, data)
            viewer.sync()

            if now - last_hud >= 0.5:
                last_hud = now
                m = env.metrics()
                warns = [env.drones[i].id for i in range(env.num_drones) if env._sensing[i]["colliding"]]
                prox = [env.drones[i].id for i in range(env.num_drones) if env._sensing[i]["prox_alert"]]
                print(f"t={m['mission_time_s']:6.1f}s | disc {m['discovered']:2d}/{m['discover_targets']} "
                      f"conf {m['confirmed']:2d}/{m['confirm_targets']} | cov {m['coverage']:6.1%} | "
                      f"coll {m['collision_events']:3d} | shares {m.get('comm_share_events', 0):3d} | "
                      f"COLLISION: {warns if warns else 'no'} | SEP<10m: {prox if prox else 'ok'}")

            elapsed = time.time() - now
            if elapsed < 0.0166:
                time.sleep(0.0166 - elapsed)
    print("[demo] closed.")


if __name__ == "__main__":
    main()