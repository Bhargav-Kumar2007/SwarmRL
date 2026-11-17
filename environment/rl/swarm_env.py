"""Custom lightweight multi-agent environment (no Gymnasium / PettingZoo).

Interface:
    reset(seed=None) -> (obs, info)
    step(actions)    -> (obs, rewards, terminated, truncated, info)

Actions are (num_drones, 3) continuous in [-1, 1]. Displacement is
action * speed * dt with diagonal normalization, applied through the existing
Drone.move() and its boundary clamping -- RL decides WHERE drones move, never
how motors behave. All sensing reuses the existing simulator functions
unchanged. MuJoCo is never imported here: training is fully headless.
"""
from dataclasses import dataclass, field
import numpy as np

from drone_model import Drone, get_nearby_drones, get_nearest_obstacle, check_collisions
from thermal import get_thermal_contacts

from .world import StaticWorld, Arena
from .mission import MissionState, DroneKnowledge
from .communication import share_knowledge
from .rewards import RewardCalculator, RewardWeights
from .observations import (OBS_DIM, GLOBAL_STATE_DIM,
                           build_observation, build_global_state)


@dataclass
class EnvConfig:
    num_drones: int = 2
    speed: float = 10.0            # existing Drone speed configuration (m/s)
    dt: float = 0.2                # kinematic sim-seconds per env step
    mission_seconds: float = 240.0
    comm_interval: int = 5         # knowledge sharing every N steps
    min_separation: float = 10.0   # existing HUD separation threshold (m)
    cell_size: float = 25.0        # exploration grid resolution (m)
    spawn_clearance: float = 12.0
    spawn_separation: float = 25.0
    share_faint: bool = False      # share only confirmed victims by default
    share_maps: bool = True        # merge explored-cell maps over comms
    max_drones: int = 50
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    @property
    def max_steps(self):
        return max(1, int(round(self.mission_seconds / self.dt)))


class SwarmEnv:
    obs_dim = OBS_DIM
    global_state_dim = GLOBAL_STATE_DIM

    def __init__(self, world: StaticWorld, config: EnvConfig | None = None):
        self.world = world
        self.cfg = config if config is not None else EnvConfig()
        if not 1 <= self.cfg.num_drones <= self.cfg.max_drones:
            raise ValueError("num_drones must be in [1, max_drones]")
        self.num_drones = self.cfg.num_drones
        self.obstacles = world.obstacles
        self.matrix = world.obstacle_positions_matrix
        self.radii = world.obstacle_bounding_radii
        self.mission = MissionState(world, self.cfg.cell_size)
        self.reward_calc = RewardCalculator(self.cfg.reward_weights)
        a = world.arena
        self.drones = [
            Drone(id=f"D{i+1:02d}", speed=self.cfg.speed, comm_radius=100.0,
                  collision_radius=1.8, active=True, mocap_id=i,
                  x_bounds=a.x_bounds, y_bounds=a.y_bounds, z_bounds=a.z_bounds)
            for i in range(self.num_drones)
        ]
        self._rng = np.random.default_rng(0)
        self.humans = []
        self.knowledge = []
        self.step_count = 0
        self._sensing = [None] * self.num_drones
        self._last_cell = [None] * self.num_drones
        self._was_colliding = [False] * self.num_drones
        self._ep_reward = np.zeros(self.num_drones)
        self._ep_components = {}
        self._state = np.zeros(GLOBAL_STATE_DIM, dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.humans = self.world.fresh_humans()
        self.mission.reset()
        self.knowledge = [DroneKnowledge(self.mission.grid_shape) for _ in self.drones]
        self.step_count = 0
        self._ep_reward = np.zeros(self.num_drones)
        self._ep_components = {}
        self._was_colliding = [False] * self.num_drones
        for d, p in zip(self.drones, self._sample_spawns()):
            d.set_position(float(p[0]), float(p[1]), float(p[2]))
            d.active = True
        self._last_cell = [self.mission.cell_of(d.pos) for d in self.drones]
        for k, d in zip(self.knowledge, self.drones):
            self.mission.mark_current_cell(k, d.pos)
        self._sense_all(np.zeros((self.num_drones, 3)))
        # Initial contacts at the spawn point are free (no reward) but populate
        # knowledge so novelty rewards stay consistent afterwards.
        for i, (k, d) in enumerate(zip(self.knowledge, self.drones)):
            self.mission.update_contacts(k, d, self._sensing[i]["contacts"], 0)
        obs = self._build_all_obs()
        self._refresh_state(obs)
        return obs, {"metrics": self.metrics()}

    def _sample_spawns(self):
        a = self.world.arena
        x0, x1 = a.x_bounds
        y0, y1 = a.y_bounds
        positions = []
        for i in range(self.num_drones):
            placed = False
            for _ in range(250):
                p = np.array([self._rng.uniform(x0 + 15.0, x1 - 15.0),
                              self._rng.uniform(y0 + 15.0, y1 - 15.0),
                              self._rng.uniform(6.0, 30.0)])
                d = np.linalg.norm(self.matrix - p, axis=1) - self.radii
                cand = np.flatnonzero(d < self.cfg.spawn_clearance)
                if any(self.obstacles[int(j)].surface_distance(p) <= self.cfg.spawn_clearance
                       for j in cand):
                    continue
                if any(np.linalg.norm(p - q) < self.cfg.spawn_separation for q in positions):
                    continue
                positions.append(p)
                placed = True
                break
            if not placed:
                ang = 2.0 * np.pi * i / self.num_drones
                positions.append(np.array([a.center_x + 40.0 * np.cos(ang),
                                           a.center_y + 40.0 * np.sin(ang),
                                           60.0 + i]))
        return positions

    def _sense_all(self, move_fracs):
        for i, d in enumerate(self.drones):
            nearest_obs, odist, orel = get_nearest_obstacle(d, self.obstacles, self.matrix)
            colliding, cname, prox, proxid = check_collisions(
                d, self.obstacles, self.drones, self.cfg.min_separation,
                self.matrix, nearest_obs, odist)
            contacts = get_thermal_contacts(d, self.humans, self.world.fires,
                                            self.obstacles, self.world.local_ambient_temp)
            nearby = get_nearby_drones(d, self.drones)
            ctype = None
            if colliding:
                ctype = ("terrain" if cname == "Terrain"
                         else (nearest_obs.obs_type if nearest_obs is not None else "building"))
            self._sensing[i] = {
                "nearest_obs": nearest_obs, "obs_dist": odist, "obs_rel": orel,
                "colliding": bool(colliding), "collision_name": cname,
                "collision_type": ctype, "prox_alert": bool(prox), "prox_drone": proxid,
                "contacts": contacts, "nearby": nearby,
                "last_move_frac": move_fracs[i],
                "cell": self.mission.cell_of(d.pos),
            }

    def _build_all_obs(self):
        return np.stack([
            build_observation(d, self.mission, k, sen, self.step_count,
                              self.cfg.max_steps, self.world.arena)
            for d, k, sen in zip(self.drones, self.knowledge, self._sensing)
        ])

    def _refresh_state(self, obs):
        self._state = build_global_state(
            self.mission, self.drones, [obs[i] for i in range(self.num_drones)],
            self._sensing, self.step_count, self.cfg.max_steps, self.cfg.max_drones)

    def global_state(self):
        return self._state

    def step(self, actions):
        a = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        a = a.reshape(self.num_drones, 3)
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        a = a / np.maximum(norms, 1.0)      # diagonal normalization: speed cap
        max_step = self.cfg.speed * self.cfg.dt
        move_fracs = np.zeros((self.num_drones, 3))
        for i, d in enumerate(self.drones):
            prev = d.pos.copy()
            d.move(a[i] * max_step)          # existing kinematic move + bounds
            move_fracs[i] = (d.pos - prev) / max_step
        self.step_count += 1
        self._sense_all(move_fracs)

        events_list = []
        for i, (k, d) in enumerate(zip(self.knowledge, self.drones)):
            sen = self._sensing[i]
            if sen["colliding"]:
                self.mission.collision_steps += 1
                if not self._was_colliding[i]:
                    self.mission.collision_events += 1
            self._was_colliding[i] = sen["colliding"]
            if sen["prox_alert"]:
                self.mission.proximity_steps += 1
            cell = self.mission.cell_of(d.pos)
            changed = cell != self._last_cell[i]
            if changed:
                team_new, self_new = self.mission.enter_cell(k, d.pos)
                self._last_cell[i] = cell
            else:
                team_new = self_new = False
            ev = self.mission.update_contacts(k, d, sen["contacts"], self.step_count)
            events_list.append((ev, team_new, self_new, changed))

        if self.step_count % self.cfg.comm_interval == 0:
            share_knowledge(self.drones, self.knowledge,
                            [s["nearby"] for s in self._sensing], self.mission,
                            self.cfg.share_faint, self.cfg.share_maps)

        indiv, comps = [], []
        for i, (ev, tn, sn, ch) in enumerate(events_list):
            sen = self._sensing[i]
            r, c = self.reward_calc.individual(
                ev, colliding=sen["colliding"], collision_type=sen["collision_type"],
                prox_alert=sen["prox_alert"],
                fire_alerts=sen["contacts"]["fire_alerts"],
                moved_frac=float(np.linalg.norm(move_fracs[i])),
                team_new_cell=tn, self_new_cell=sn, cell_changed=ch)
            indiv.append(r)
            comps.append(c)
        totals, team_pool, bonus = self.reward_calc.combine(indiv, comps)
        self._ep_reward += np.asarray(totals)
        for c in comps:
            for kk, vv in c.items():
                self._ep_components[kk] = self._ep_components.get(kk, 0.0) + vv
        self._ep_components["team_bonus"] = (
            self._ep_components.get("team_bonus", 0.0) + bonus * self.num_drones)

        obs = self._build_all_obs()
        self._refresh_state(obs)
        terminated = self.mission.all_confirmed
        truncated = self.step_count >= self.cfg.max_steps
        info = {
            "metrics": self.metrics(),
            "step_components": {kk: float(np.mean([c[kk] for c in comps])) for kk in comps[0]},
            "team_bonus": bonus,
        }
        return obs, np.asarray(totals, dtype=np.float32), terminated, truncated, info

    def metrics(self):
        m = self.mission
        return {
            "steps": self.step_count,
            "mission_time_s": self.step_count * self.cfg.dt,
            "discovered": len(m.team_discovered),
            "confirmed": len(m.team_confirmed),
            "discover_targets": len(m.discover_targets),
            "confirm_targets": len(m.confirm_targets),
            "discovered_fraction": m.discovered_fraction,
            "confirmed_fraction": m.confirmed_fraction,
            "coverage": m.coverage,
            "collision_events": m.collision_events,
            "collision_steps": m.collision_steps,
            "proximity_steps": m.proximity_steps,
            "cell_entries": m.cell_entries,
            "redundant_fraction": (m.redundant_entries / m.cell_entries) if m.cell_entries else 0.0,
            "comm_share_events": m.comm_share_events,
            "episode_reward_mean": float(self._ep_reward.mean()),
            "episode_reward_per_drone": self._ep_reward.tolist(),
            "reward_components": dict(self._ep_components),
        }


class VecSwarmEnv:
    """Synchronous parallel environments sharing one StaticWorld.

    Same drone count in every sub-env (curriculum stages rebuild the vec env).
    Auto-resets on termination/truncation and reports final-episode metrics.
    Seeding is deterministic: base_seed + 100003 * env_index + episode_count.
    """

    def __init__(self, world: StaticWorld, config: EnvConfig,
                 num_envs: int = 1, base_seed: int = 0):
        self.envs = [SwarmEnv(world, config) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.num_drones = config.num_drones
        self.obs_dim = OBS_DIM
        self.global_state_dim = GLOBAL_STATE_DIM
        self.base_seed = base_seed
        self._ep_counts = [0] * num_envs

    def _seed_for(self, i):
        return self.base_seed + 100003 * i + self._ep_counts[i]

    def reset(self, seed=None):
        if seed is not None:
            self.base_seed = int(seed)
            self._ep_counts = [0] * self.num_envs
        obs, states, infos = [], [], []
        for i, env in enumerate(self.envs):
            o, info = env.reset(seed=self._seed_for(i))
            obs.append(o)
            states.append(env.global_state())
            infos.append(info)
        return np.stack(obs), np.stack(states), infos

    def step(self, actions):
        actions = np.asarray(actions)
        obs_l, rew_l, done_l, state_l, info_l, finals = [], [], [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(actions[i])
            done = bool(term or trunc)
            if done:
                fm = dict(info["metrics"])
                fm["terminated"] = bool(term)
                fm["env_index"] = i
                finals.append(fm)
                self._ep_counts[i] += 1
                o, _ = env.reset(seed=self._seed_for(i))
            obs_l.append(o)
            rew_l.append(r)
            done_l.append(done)
            state_l.append(env.global_state())
            info_l.append(info)
        return (np.stack(obs_l), np.stack(rew_l), np.asarray(done_l, dtype=bool),
                np.stack(state_l), info_l, finals)