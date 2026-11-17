"""Static world construction shared by every training environment.

The procedural city is generated ONCE per process with the existing
ProceduralEnvironmentGenerator. Obstacles / fires / cold zones are read-only and
shared across parallel envs; humans are mutable (thermal.py sets detection
flags), so each env gets fresh per-episode copies. The generated XML string is
kept only so the MuJoCo demo can compile the identical scene later.
"""
from dataclasses import dataclass, replace
from typing import List, Optional
import numpy as np
from environment_generator import ProceduralEnvironmentGenerator
from drone_model import Obstacle


@dataclass
class Arena:
    """Training sub-region of the 10 km city (existing Drone bounds are set to it).

    Default center is the generator's 'Disaster Ground Zero' rubble sector.
    """
    center_x: float = 260.0
    center_y: float = 160.0
    size: float = 800.0
    z_min: float = 0.2
    z_max: float = 150.0

    @property
    def x_bounds(self):
        return (self.center_x - self.size / 2.0, self.center_x + self.size / 2.0)

    @property
    def y_bounds(self):
        return (self.center_y - self.size / 2.0, self.center_y + self.size / 2.0)

    @property
    def z_bounds(self):
        return (self.z_min, self.z_max)

    def contains_xy(self, pos) -> bool:
        x0, x1 = self.x_bounds; y0, y1 = self.y_bounds
        return bool(x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1)


def _bounding_radius(o) -> float:
    if o.shape == "box":
        return float(np.linalg.norm(o.size_params[:3] / 2.0))
    if o.shape == "cylinder":
        return float(np.hypot(o.size_params[0], o.size_params[1] / 2.0))
    if o.shape == "sphere":
        return float(o.size_params[0])
    if o.shape == "tree":
        return float(o.size_params[3] + o.size_params[2])
    return float(np.max(o.size_params))


class StaticWorld:
    """Generate-once container for all static simulation data."""

    def __init__(self, seed=12345, env_size=10000.0, num_buildings=800,
                 num_rubble_zones=50, num_rubble_per_zone=30, num_trees=900,
                 num_poles=350, num_humans=150, arena: Optional[Arena] = None):
        self.seed = seed
        self.arena = arena if arena is not None else Arena()
        gen = ProceduralEnvironmentGenerator(
            env_size=env_size, num_buildings=num_buildings,
            num_rubble_zones=num_rubble_zones, num_rubble_per_zone=num_rubble_per_zone,
            num_trees=num_trees, num_poles=num_poles, seed=seed, num_humans=num_humans)
        # generate() builds XML + obstacle metadata; nothing is written to disk.
        self.xml, gen_obstacles = gen.generate()
        self.obstacles: List[Obstacle] = [
            Obstacle(name=g.name, obs_type=g.obs_type, pos=g.pos, shape=g.shape,
                     size_params=g.size_params, heading_deg=g.heading_deg)
            for g in gen_obstacles]
        self.obstacle_positions_matrix = np.array([o.pos for o in self.obstacles], dtype=np.float64)
        self.obstacle_bounding_radii = np.array([_bounding_radius(o) for o in self.obstacles])
        self.fires = gen.fires
        self.cold_zones = gen.cold_zones
        self.local_ambient_temp = gen.local_ambient_temp
        # The two plaza demonstration victims are NEVER ordinary training targets.
        self.human_templates = [h for h in gen.humans if not h.id.startswith("Human_Demo")]
        in_arena = [h for h in self.human_templates if self.arena.contains_xy(h.pos)]
        self.discover_target_ids = frozenset(h.id for h in in_arena)
        # Indoor victims sit inside solid buildings the kinematic drone cannot
        # enter; only outdoor victims are required for confirmation progress.
        # Indoor victims still count as discoveries (faint contact from above
        # the footprint is supported by the existing thermal model).
        self.confirm_target_ids = frozenset(h.id for h in in_arena if not h.indoor)

    def fresh_humans(self):
        """Per-episode mutable copies with detection flags cleared."""
        return [replace(h, pos=h.pos.copy(), detected_faint=False, confirmed=False)
                for h in self.human_templates]