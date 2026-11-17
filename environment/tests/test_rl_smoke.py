"""Smoke tests for the SwarmRL lightweight RL layer (no torch / mujoco needed).

Run from the repo root:  python tests/test_rl_smoke.py
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment"))
import numpy as np
from rl.world import StaticWorld, Arena
from rl.swarm_env import SwarmEnv, VecSwarmEnv, EnvConfig
from rl.rewards import RewardCalculator, RewardWeights
from rl.observations import OBS_DIM, GLOBAL_STATE_DIM

PASS = []
def check(name, cond):
    assert cond, "FAILED: " + name
    PASS.append(name); print("  [PASS]", name)

world = StaticWorld(seed=12345)
print("world: %d obstacles, %d discover targets, %d confirm targets" % (
    len(world.obstacles), len(world.discover_target_ids), len(world.confirm_target_ids)))
check("demo victims excluded", all(not h.id.startswith("Human_Demo") for h in world.human_templates))
check("targets exist in arena", len(world.confirm_target_ids) > 0)

for n in (2, 5, 10, 25, 50):
    e = SwarmEnv(world, EnvConfig(num_drones=n))
    o, _ = e.reset(seed=n)
    assert o.shape == (n, OBS_DIM) and np.isfinite(o).all()
    assert e.global_state().shape == (GLOBAL_STATE_DIM,) and np.isfinite(e.global_state()).all()
check("fixed obs/state dims for N=2,5,10,25,50", True)

e = SwarmEnv(world, EnvConfig(num_drones=2)); e.reset(seed=7)
p0 = e.drones[0].pos.copy()
e.step(np.array([[1., 1., 1.], [0., 0., 0.]]))
check("diagonal action capped at speed*dt", abs(np.linalg.norm(e.drones[0].pos - p0) - 2.0) < 1e-9)
p1 = e.drones[0].pos.copy()
e.step(np.array([[0.5, 0., 0.], [0., 0., 0.]]))
check("half action -> half step", abs(np.linalg.norm(e.drones[0].pos - p1) - 1.0) < 1e-9)
e.drones[0].set_position(world.arena.x_bounds[1] - 0.5, 160., 20.)
e.step(np.array([[1., 0., 0.], [0., 0., 0.]]))
check("arena boundary clamped", e.drones[0].pos[0] <= world.arena.x_bounds[1] + 1e-9)

e1 = SwarmEnv(world, EnvConfig(num_drones=2)); e2 = SwarmEnv(world, EnvConfig(num_drones=2))
o1, _ = e1.reset(seed=42); o2, _ = e2.reset(seed=42)
r = np.random.default_rng(123); same = np.array_equal(o1, o2)
for _ in range(30):
    a = r.uniform(-1, 1, (2, 3))
    o1, r1, _, _, _ = e1.step(a); o2, r2, _, _, _ = e2.step(a)
    same &= np.array_equal(o1, o2) and np.array_equal(r1, r2)
check("seeded determinism (obs + rewards)", same)

check("pre-reset has progress", e1.mission.cell_entries > 0)
e1.reset(seed=1)
check("reset clears mission/knowledge/flags", (
    len(e1.mission.team_discovered) == 0 and len(e1.mission.team_confirmed) == 0
    and e1.mission.collision_events == 0 and e1.step_count == 0
    and all(len(k.faint) == 0 and len(k.confirmed) == 0 for k in e1.knowledge)
    and all(not h.detected_faint and not h.confirmed for h in e1.humans)))

e = SwarmEnv(world, EnvConfig(num_drones=3)); e.reset(seed=11)
r = np.random.default_rng(5)
for _ in range(40): e.step(r.uniform(-1, 1, (3, 3)))
obs_before = e._build_all_obs().copy()
known = set()
for k in e.knowledge: known |= set(k.faint) | k.confirmed
moved = 0
for h in e.humans:
    if h.id not in known:
        h.pos = h.pos + np.array([20000., 20000., 0.]); moved += 1
obs_after = e._build_all_obs()
check("obs invariant to undiscovered victim positions (%d moved)" % moved,
      np.array_equal(obs_before, obs_after))

from thermal import get_thermal_contacts
from drone_model import Drone as SimDrone
probe = SimDrone(id="probe")
target, tpos = None, None
for h in world.human_templates:
    if h.id not in world.confirm_target_ids: continue
    p = h.pos + np.array([3., 0., 0.5])
    probe.set_position(*p)
    res = get_thermal_contacts(probe, [h], [], world.obstacles, world.local_ambient_temp)
    if any(c["id"] == h.id for c in res["confirmed"]):
        target, tpos = h, p; break
assert target is not None, "no confirmable target found"
e = SwarmEnv(world, EnvConfig(num_drones=2, comm_interval=2)); e.reset(seed=3)
e.drones[0].set_position(*tpos)
e.drones[1].set_position(tpos[0] + 60., tpos[1], tpos[2])
for _ in range(4): e.step(np.zeros((2, 3)))
check("drone0 confirmed victim", target.id in e.knowledge[0].confirmed)
check("teammate received confirmation over comms", target.id in e.knowledge[1].confirmed)
e.reset(seed=3)
e.drones[0].set_position(*tpos)
e.drones[1].set_position(tpos[0] + 240., tpos[1], tpos[2])
for _ in range(4): e.step(np.zeros((2, 3)))
check("no sharing outside comm radius", target.id not in e.knowledge[1].confirmed)

bld = next(o for o in world.obstacles if o.obs_type == "Building" and world.arena.contains_xy(o.pos))
e = SwarmEnv(world, EnvConfig(num_drones=2)); e.reset(seed=9)
e.drones[0].set_position(*bld.pos)
_, rew, _, _, info = e.step(np.zeros((2, 3)))
check("building collision penalized", rew[0] < 0 and info["step_components"]["collision"] < 0)
check("collision event counted", e.mission.collision_events >= 1)

rc = RewardCalculator(RewardWeights())
ev = {"personal_new_faint": 1, "team_discoveries": 1, "team_confirmations": 1, "coop_confirmations": 1}
r_ind, comp = rc.individual(ev, colliding=False, collision_type=None, prox_alert=False,
                            fire_alerts=[], moved_frac=1.0, team_new_cell=True,
                            self_new_cell=True, cell_changed=True)
check("components sum to individual reward", abs(sum(comp.values()) - r_ind) < 1e-9)
totals, pool, bonus = rc.combine([r_ind, 0.0], [comp, {k: 0.0 for k in comp}])
check("team broadcast math", abs(totals[0] - (r_ind + bonus)) < 1e-9 and bonus == 0.5 * pool)

v = VecSwarmEnv(world, EnvConfig(num_drones=2, mission_seconds=2.0), num_envs=2, base_seed=100)
o, s, _ = v.reset(seed=100)
finals = []
for _ in range(10):
    o, r, d, s, infos, f = v.step(np.zeros((2, 2, 3))); finals += f
check("vec env truncation + auto-reset", d.all() and len(finals) == 2 and o.shape == (2, 2, OBS_DIM))
check("final metrics reported", all("episode_reward_mean" in fm for fm in finals))

import xml.etree.ElementTree as ET
xml = world.xml
start = xml.find('<body name="D03" mocap="true"')
end = xml.find("</body>", start) + len("</body>")
block = xml[start:end]
extra = []
for i in range(4, 51):
    b = block.replace("D03", "D%02d" % i)
    b = "\n".join(l for l in b.split("\n") if "comm_sphere" not in l)
    extra.append(b)
inj = xml[:end] + "\n" + "\n".join(extra) + xml[end:]
root = ET.fromstring(inj)
nmocap = sum(1 for b in root.iter("body") if b.get("mocap") == "true")
check("demo XML well-formed with 50 mocap drones", nmocap == 50)

print("\nALL %d SMOKE TESTS PASSED" % len(PASS))

print("\nThroughput benchmark (random actions, headless):")
for n in (2, 10, 50):
    e = SwarmEnv(world, EnvConfig(num_drones=n)); e.reset(seed=1)
    r = np.random.default_rng(0)
    steps = 200
    t = time.perf_counter()
    for _ in range(steps): e.step(r.uniform(-1, 1, (n, 3)))
    dt = time.perf_counter() - t
    print("  N=%2d: %7.1f env steps/s | %8.1f agent steps/s | %.2f ms/step" % (
        n, steps / dt, steps * n / dt, dt / steps * 1000))