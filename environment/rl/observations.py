"""Fixed-size observation and centralized-state builders.

Actor observation: 41 floats, independent of swarm size, LOCAL sensors only.
Undiscovered victim coordinates never appear here.

Layout (OBS_DIM = 41):
  0:3   position normalized to arena [-1, 1]
  3:6   last displacement / max step (movement direction estimate)
  6     remaining mission time fraction
  7     nearest obstacle surface distance / 100 m, clipped [0, 1]
  8:11  nearest obstacle unit direction
  11:15 obstacle type one-hot (Building, Rubble, Tree, Pole)
  15    faint human contact count / 5, clipped
  16    nearest faint human distance / 50 m, clipped (1 = none)
  17:20 nearest faint human unit direction (0 = none)
  20    confirmed humans in sensor range / 5, clipped
  21    fire alert count / 3, clipped
  22    nearest fire alert distance / 25 m, clipped (1 = none)
  23    nearby drone count / 10, clipped
  24:36 nearest 3 drones: relative position / comm_radius (3 each) + valid flag
  36    communication available (>= 1 neighbor)
  37    current cell already visited by this drone
  38    local exploration coverage (5x5 cell window, team map)
  39    team victim discovery progress fraction
  40    team victim confirmation progress fraction

Centralized critic state (GLOBAL_STATE_DIM = 94) = 12 mission scalars +
mean-pooled obs (41) + max-pooled obs (41). Fixed size for any drone count.
"""
import numpy as np

OBS_DIM = 41
_N_GLOBAL_SCALARS = 12
GLOBAL_STATE_DIM = _N_GLOBAL_SCALARS + 2 * OBS_DIM

_OBS_TYPES = ("Building", "Rubble", "Tree", "Pole")
_OBS_RANGE = 100.0
_FAINT_RANGE = 50.0    # matches thermal.FAINT_SIGNAL_RADIUS
_FIRE_RANGE = 25.0     # matches thermal.FIRE_ALERT_RADIUS
_K_NEIGHBORS = 3


def _unit(vec):
    v = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3)


def build_observation(drone, mission, knowledge, sensing, step, max_steps, arena):
    """Assemble one drone's observation from already-computed sensor outputs."""
    o = np.zeros(OBS_DIM, dtype=np.float32)
    half = arena.size / 2.0
    o[0] = (drone.pos[0] - arena.center_x) / half
    o[1] = (drone.pos[1] - arena.center_y) / half
    o[2] = (drone.pos[2] - arena.z_min) / max(arena.z_max - arena.z_min, 1e-6) * 2.0 - 1.0
    o[3:6] = sensing["last_move_frac"]
    o[6] = 1.0 - step / max(max_steps, 1)

    o[7] = min(sensing["obs_dist"] / _OBS_RANGE, 1.0)
    o[8:11] = _unit(sensing["obs_rel"])
    obs = sensing["nearest_obs"]
    if obs is not None and obs.obs_type in _OBS_TYPES:
        o[11 + _OBS_TYPES.index(obs.obs_type)] = 1.0

    contacts = sensing["contacts"]
    faint_humans = [c for c in contacts["faint"] if c.get("type") != "fire"]
    o[15] = min(len(faint_humans) / 5.0, 1.0)
    if faint_humans:
        nearest = faint_humans[0]  # distance-sorted by thermal.py
        o[16] = min(nearest["distance"] / _FAINT_RANGE, 1.0)
        o[17:20] = _unit(nearest["relative_position"])
    else:
        o[16] = 1.0
    o[20] = min(len(contacts["confirmed"]) / 5.0, 1.0)
    alerts = contacts["fire_alerts"]
    o[21] = min(len(alerts) / 3.0, 1.0)
    o[22] = min(alerts[0]["distance"] / _FIRE_RANGE, 1.0) if alerts else 1.0

    nearby = sensing["nearby"]
    o[23] = min(len(nearby) / 10.0, 1.0)
    for k in range(_K_NEIGHBORS):
        base = 24 + 4 * k
        if k < len(nearby):
            rel = np.asarray(nearby[k]["relative_position"]) / max(drone.comm_radius, 1e-6)
            o[base:base + 3] = np.clip(rel, -1.0, 1.0)
            o[base + 3] = 1.0
    o[36] = 1.0 if nearby else 0.0

    ix, iy = sensing["cell"]
    o[37] = 1.0 if knowledge.visited[ix, iy] else 0.0
    x0, x1 = max(ix - 2, 0), min(ix + 3, mission.nx)
    y0, y1 = max(iy - 2, 0), min(iy + 3, mission.ny)
    o[38] = float(mission.team_visited[x0:x1, y0:y1].mean())
    o[39] = mission.discovered_fraction
    o[40] = mission.confirmed_fraction
    return o


def build_global_state(mission, drones, all_obs, sensing_list, step, max_steps,
                       max_drones=50):
    """Centralized training-only state (fixed size, any swarm size)."""
    n = len(drones)
    s = np.zeros(_N_GLOBAL_SCALARS, dtype=np.float32)
    s[0] = 1.0 - step / max(max_steps, 1)
    s[1] = n / float(max_drones)
    s[2] = mission.discovered_fraction
    s[3] = mission.confirmed_fraction
    s[4] = mission.coverage
    s[5] = float(np.mean([o[38] for o in all_obs]))
    s[6] = float(np.mean([o[7] for o in all_obs]))
    s[7] = float(np.mean([sen["colliding"] for sen in sensing_list]))
    s[8] = float(np.mean([sen["prox_alert"] for sen in sensing_list]))
    s[9] = float(np.mean([o[23] for o in all_obs]))
    s[10] = float(np.mean([o[15] for o in all_obs]))
    s[11] = float(np.mean([len(sen["contacts"]["fire_alerts"]) > 0 for sen in sensing_list]))
    stacked = np.stack(all_obs)
    return np.concatenate([s, stacked.mean(axis=0), stacked.max(axis=0)]).astype(np.float32)