"""
Disaster-Response 3D Drone Prototype with Congested 10 km x 10 km Procedural City.
Features dense urban districts, skyscrapers, street canyons, alleyways, heavy rubble, trees, and poles.
"""

import sys
import os
import time
import math
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
import mujoco
import mujoco.viewer

from environment_generator import ProceduralEnvironmentGenerator
from drone_model import (
    Drone,
    Obstacle,
    get_nearby_drones,
    get_nearest_obstacle,
    check_collisions,
)
from gpu_utils import (
    setup_high_performance_gpu,
    enable_high_resolution_timer,
    disable_high_resolution_timer,
)

# =============================================================================
# ENVIRONMENT GENERATION CONFIGURATION (Easily manually changeable)
# =============================================================================
ENVIRONMENT_SIZE = 10000.0       # 10 km x 10 km square (-5,000m to +5,000m)
ENVIRONMENT_SEED = 12345         # Deterministic random seed (change manually or via --seed)

# Congested Density Counts
NUM_BUILDINGS = 800              # Skyscrapers, high-rises, mid-rises, dense row-houses
NUM_RUBBLE_ZONES = 50            # Disaster rubble clusters (heavy concentration in disaster zone)
NUM_RUBBLE_PER_ZONE = 30         # Pieces per cluster (Total: 1,500 rubble pieces)
NUM_TREES = 900                  # Urban parks, street avenues, forest patches
NUM_POLES = 350                  # Streetlights, utility poles, telecom towers

SAFE_STARTING_RADIUS = 100.0     # 200m x 200m clear takeoff plaza at origin (0, 0)
# =============================================================================

# Detect OS and initialize Windows console ANSI VT100 mode
IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    import ctypes
    os.system('')  # Enables ANSI escape processing in Windows console

    def is_key_pressed(vk_code: int) -> bool:
        """Returns True if the specified Virtual-Key is currently held down."""
        try:
            return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
        except Exception:
            return False
else:
    def is_key_pressed(vk_code: int) -> bool:
        return False

# Virtual Key Codes (Windows)
VK_TAB = 0x09
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_A = 0x41
VK_C = 0x43
VK_D = 0x44
VK_E = 0x45
VK_F = 0x46
VK_M = 0x4D
VK_N = 0x4E
VK_P = 0x50
VK_Q = 0x51
VK_R = 0x52
VK_S = 0x53
VK_W = 0x57
VK_Z = 0x5A


class DroneSimulator:
    """Manages the congested 10 km MuJoCo disaster environment, flight controls, and HUD."""

    def __init__(
        self,
        scene_path: str = "generated_scene.xml",
        env_size: float = ENVIRONMENT_SIZE,
        num_buildings: int = NUM_BUILDINGS,
        num_rubble_zones: int = NUM_RUBBLE_ZONES,
        num_rubble_per_zone: int = NUM_RUBBLE_PER_ZONE,
        num_trees: int = NUM_TREES,
        num_poles: int = NUM_POLES,
        seed: int = ENVIRONMENT_SEED,
        safe_radius: float = SAFE_STARTING_RADIUS,
    ):
        self.scene_path = scene_path
        self.seed = seed

        # Configure discrete GPU acceleration (NVIDIA RTX 4050)
        self.gpu_info = setup_high_performance_gpu()
        if self.gpu_info.get("is_discrete"):
            print(f"[Init] Hardware Acceleration: {self.gpu_info['renderer']} (High-Performance Discrete GPU)")
        else:
            print(f"[Init] Renderer: {self.gpu_info.get('renderer', 'OpenGL Hardware Accelerated')}")

        # 1. Procedural Generation: Generate congested city environment
        print(f"[Init] Procedurally generating congested 10 km disaster city (Seed: {seed})...")
        generator = ProceduralEnvironmentGenerator(
            env_size=env_size,
            num_buildings=num_buildings,
            num_rubble_zones=num_rubble_zones,
            num_rubble_per_zone=num_rubble_per_zone,
            num_trees=num_trees,
            num_poles=num_poles,
            seed=seed,
            safe_radius=safe_radius,
        )
        gen_obstacles = generator.export_to_file(self.scene_path)

        # Convert to runtime Obstacle objects for detection
        self.obstacles: List[Obstacle] = []
        for go in gen_obstacles:
            self.obstacles.append(
                Obstacle(
                    name=go.name,
                    obs_type=go.obs_type,
                    pos=go.pos,
                    shape=go.shape,
                    size_params=go.size_params,
                    heading_deg=go.heading_deg,
                )
            )

        # Precompute positions matrix for fast vectorized distance queries (<0.02ms)
        self.obstacle_positions_matrix = np.array([obs.pos for obs in self.obstacles], dtype=np.float64)

        # 2. Load MuJoCo Model
        print(f"[Init] Loading MuJoCo model from: {self.scene_path}")
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)

        # Optimize shadow quality for silky-smooth 60+ FPS rendering on GPU
        if hasattr(self.model, "vis") and hasattr(self.model.vis, "quality"):
            self.model.vis.quality.shadowsize = min(self.model.vis.quality.shadowsize, 2048)

        # 3. Initialize Drones
        # Primary Drone D01: Starts at origin (0, 0, 5) inside the safe starting zone
        self.drone_d01 = Drone(
            id="D01",
            pos=np.array([0.0, 0.0, 5.0], dtype=np.float64),
            speed=10.0,
            comm_radius=100.0,
            collision_radius=1.8,
            active=True,
            mocap_id=0,
            x_bounds=(-env_size / 2.0, env_size / 2.0),
            y_bounds=(-env_size / 2.0, env_size / 2.0),
            z_bounds=(0.2, 1000.0),
        )

        # Standby Drone D02: Spawnable at (42, -51, 8) within 100m range
        self.d02_spawn_pos = np.array([42.0, -51.0, 8.0], dtype=np.float64)
        self.drone_d02 = Drone(
            id="D02",
            pos=np.array([0.0, 0.0, -100.0], dtype=np.float64),
            speed=10.0,
            comm_radius=100.0,
            collision_radius=1.8,
            active=False,
            mocap_id=1,
            x_bounds=(-env_size / 2.0, env_size / 2.0),
            y_bounds=(-env_size / 2.0, env_size / 2.0),
            z_bounds=(0.2, 1000.0),
        )

        # Standby Drone D03: Spawnable at (-65, 55, 12)
        self.d03_spawn_pos = np.array([-65.0, 55.0, 12.0], dtype=np.float64)
        self.drone_d03 = Drone(
            id="D03",
            pos=np.array([0.0, 0.0, -100.0], dtype=np.float64),
            speed=10.0,
            comm_radius=100.0,
            collision_radius=1.8,
            active=False,
            mocap_id=2,
            x_bounds=(-env_size / 2.0, env_size / 2.0),
            y_bounds=(-env_size / 2.0, env_size / 2.0),
            z_bounds=(0.2, 1000.0),
        )

        self.drones = [self.drone_d01, self.drone_d02, self.drone_d03]
        self.active_pilot_idx = 0

        # Collision monitoring state
        self.total_collisions = 0
        self.was_colliding = False

        # Speed boost toggle / multiplier for rapid traversal across 10km
        self.sprint_multiplier = 1.0

        # Fallback impulses from MuJoCo GLFW key_callback
        self.callback_impulses = np.zeros(3, dtype=np.float64)
        self.key_toggles = {"spawn_d02": False, "spawn_d03": False, "switch_drone": False, "reset": False}
        self.prev_key_states: Dict[int, bool] = {}

        # Set initial positions in MuJoCo mocap data
        self._sync_mocap()
        mujoco.mj_forward(self.model, self.data)

    @property
    def current_drone(self) -> Drone:
        return self.drones[self.active_pilot_idx]

    def _sync_mocap(self) -> None:
        """Syncs Python drone positions to MuJoCo mocap bodies."""
        for drone in self.drones:
            if drone.mocap_id < self.model.nmocap:
                self.data.mocap_pos[drone.mocap_id] = drone.pos

    def toggle_drone_d02(self) -> None:
        """Spawns or despawns D02 in the scene for 100m range testing."""
        if not self.drone_d02.active:
            self.drone_d02.pos = self.d02_spawn_pos.copy()
            self.drone_d02.active = True
            print("\n>>> [EVENT] Spawned Drone D02 at (42.0, -51.0, 8.0) <<<")
        else:
            self.drone_d02.pos = np.array([0.0, 0.0, -100.0], dtype=np.float64)
            self.drone_d02.active = False
            if self.active_pilot_idx == 1:
                self.active_pilot_idx = 0
            print("\n>>> [EVENT] Despawned Drone D02 <<<")
        self._sync_mocap()

    def toggle_drone_d03(self) -> None:
        """Spawns or despawns D03 in the scene."""
        if not self.drone_d03.active:
            self.drone_d03.pos = self.d03_spawn_pos.copy()
            self.drone_d03.active = True
            print("\n>>> [EVENT] Spawned Drone D03 at (-65.0, 55.0, 12.0) <<<")
        else:
            self.drone_d03.pos = np.array([0.0, 0.0, -100.0], dtype=np.float64)
            self.drone_d03.active = False
            if self.active_pilot_idx == 2:
                self.active_pilot_idx = 0
            print("\n>>> [EVENT] Despawned Drone D03 <<<")
        self._sync_mocap()

    def switch_active_pilot(self) -> None:
        """Cycles control between active drones."""
        active_indices = [i for i, d in enumerate(self.drones) if d.active]
        if len(active_indices) > 1:
            curr_pos_in_active = active_indices.index(self.active_pilot_idx)
            next_idx = active_indices[(curr_pos_in_active + 1) % len(active_indices)]
            self.active_pilot_idx = next_idx
            print(f"\n>>> [PILOT SWITCH] Now controlling {self.current_drone.id} <<<")

    def reset_current_drone(self) -> None:
        """Resets the currently active drone to its start position."""
        if self.current_drone.id == "D01":
            self.current_drone.set_position(0.0, 0.0, 5.0)
        elif self.current_drone.id == "D02":
            self.current_drone.set_position(42.0, -51.0, 8.0)
        elif self.current_drone.id == "D03":
            self.current_drone.set_position(-65.0, 55.0, 12.0)
        self._sync_mocap()
        print(f"\n>>> [RESET] Reset {self.current_drone.id} to origin <<<")

    def key_callback(self, keycode: int) -> None:
        """Fallback GLFW key callback from MuJoCo viewer window."""
        try:
            char = chr(keycode).upper()
        except (ValueError, OverflowError):
            char = ""

        if char == "W":
            self.callback_impulses[1] += 1.0
        elif char == "S":
            self.callback_impulses[1] -= 1.0
        elif char == "A":
            self.callback_impulses[0] -= 1.0
        elif char == "D":
            self.callback_impulses[0] += 1.0
        elif char in (" ", "E"):
            self.callback_impulses[2] += 1.0
        elif char in ("C", "Q", "Z") or keycode in (340, 344):  # Shift
            self.callback_impulses[2] -= 1.0
        elif char in ("N", "P"):
            self.key_toggles["spawn_d02"] = True
        elif char == "M":
            self.key_toggles["spawn_d03"] = True
        elif keycode == 9 or char == "\t":  # Tab
            self.key_toggles["switch_drone"] = True
        elif char == "R":
            self.key_toggles["reset"] = True

    def _poll_inputs(self, dt: float) -> np.ndarray:
        """Polls keyboard inputs via Windows GetAsyncKeyState with MuJoCo key_callback fallback."""
        move_dir = np.zeros(3, dtype=np.float64)

        if IS_WINDOWS:
            # 3-Axis movement: W/S (+/-Y), A/D (+/-X), Space/Shift (+/-Z)
            if is_key_pressed(VK_W):
                move_dir[1] += 1.0
            if is_key_pressed(VK_S):
                move_dir[1] -= 1.0
            if is_key_pressed(VK_D):
                move_dir[0] += 1.0
            if is_key_pressed(VK_A):
                move_dir[0] -= 1.0
            if is_key_pressed(VK_SPACE) or is_key_pressed(VK_E):
                move_dir[2] += 1.0
            if is_key_pressed(VK_SHIFT) or is_key_pressed(VK_C) or is_key_pressed(VK_Q):
                move_dir[2] -= 1.0

            # Sprint key (hold Ctrl or F for 5x speed boost across 10km world)
            if is_key_pressed(VK_CONTROL) or is_key_pressed(VK_F):
                self.sprint_multiplier = 5.0
            else:
                self.sprint_multiplier = 1.0

            # Single-press action buttons (debounced)
            def check_just_pressed(vk: int) -> bool:
                cur = is_key_pressed(vk)
                prev = self.prev_key_states.get(vk, False)
                self.prev_key_states[vk] = cur
                return cur and not prev

            if check_just_pressed(VK_N) or check_just_pressed(VK_P):
                self.toggle_drone_d02()
            if check_just_pressed(VK_M):
                self.toggle_drone_d03()
            if check_just_pressed(VK_TAB):
                self.switch_active_pilot()
            if check_just_pressed(VK_R):
                self.reset_current_drone()

        # Incorporate fallback impulses from GLFW callback
        if np.any(self.callback_impulses != 0):
            move_dir += self.callback_impulses
            self.callback_impulses.fill(0.0)

        # Handle queued toggles from GLFW callback
        if self.key_toggles["spawn_d02"]:
            self.toggle_drone_d02()
            self.key_toggles["spawn_d02"] = False
        if self.key_toggles["spawn_d03"]:
            self.toggle_drone_d03()
            self.key_toggles["spawn_d03"] = False
        if self.key_toggles["switch_drone"]:
            self.switch_active_pilot()
            self.key_toggles["switch_drone"] = False
        if self.key_toggles["reset"]:
            self.reset_current_drone()
            self.key_toggles["reset"] = False

        # Normalize horizontal movement for uniform speed
        horiz_norm = np.linalg.norm(move_dir[:2])
        if horiz_norm > 1.0:
            move_dir[0] /= horiz_norm
            move_dir[1] /= horiz_norm

        effective_speed = self.current_drone.speed * self.sprint_multiplier
        delta = move_dir * (effective_speed * dt)
        return delta

    def print_hud(
        self,
        drone: Drone,
        nearby_drones: List[Dict[str, Any]],
        nearest_obs: Optional[Obstacle],
        obs_dist: float,
        obs_rel: tuple,
        is_colliding: bool,
        coll_entity: Optional[str],
        prox_alert: bool,
        prox_drone: Optional[str],
    ) -> None:
        """Renders formatted terminal telemetry matching specifications with fixed line padding."""
        coll_str = f"COLLISION DETECTED ({coll_entity})" if is_colliding else "NO"
        prox_str = f"ALERT! Closer than 10m to {prox_drone}" if prox_alert else "OK (> 10m)"
        effective_speed = drone.speed * self.sprint_multiplier

        hud = [
            "--------------------------------",
            f"Drone: {drone.id}",
            f"Environment: Congested City 10 km x 10 km (Seed: {self.seed})",
            f"Active Objects: {len(self.obstacles)} (Buildings, Rubble, Trees, Poles)",
            f"GPU Acceleration: {self.gpu_info.get('renderer', 'DirectX/OpenGL Hardware')[:36]}",
            "",
            "Position:",
            f"X: {drone.pos[0]:.1f} m",
            f"Y: {drone.pos[1]:.1f} m",
            f"Z: {drone.pos[2]:.1f} m",
            "",
            f"Speed:",
            f"{effective_speed:.1f} m/s" + (" [SPRINT]" if self.sprint_multiplier > 1.0 else ""),
            "",
            f"Communication radius:",
            f"{drone.comm_radius:.0f} m",
            "",
            f"Nearby drones:",
            f"{len(nearby_drones)}",
        ]

        if nearby_drones:
            for nd in nearby_drones:
                rx, ry, rz = nd["relative_position"]
                hud.append(f"  {nd['drone_id']} | Dist: {nd['distance']:.1f} m | Rel: ({rx:.1f}, {ry:.1f}, {rz:.1f})")

        hud.extend([
            "",
            "Nearest obstacle:",
            f"{nearest_obs.name if nearest_obs else 'None'}",
            f"Distance:",
            f"{obs_dist:.1f} m",
            f"Relative position:",
            f"({obs_rel[0]:.1f}, {obs_rel[1]:.1f}, {obs_rel[2]:.1f})",
            "",
            f"Collision:",
            f"{coll_str}",
            "",
            f"Total collisions:",
            f"{self.total_collisions}",
            "",
            f"Drone Separation (<10m): {prox_str}",
            "--------------------------------",
            "Flight : W/S (+/-Y), A/D (+/-X), SPACE (+Z), SHIFT/C/Q (-Z)",
            "Sprint : Hold CTRL or F for 5x Speed Boost across 10 km map",
            "Actions: N/P (Spawn D02), M (Spawn D03), TAB (Switch Pilot), R (Reset), ESC (Quit)",
        ])

        # Pad every line to 72 characters so it completely overwrites previous console text
        padded_lines = [f"{line:<72}" for line in hud]
        sys.stdout.write("\033[H" + "\n".join(padded_lines) + "\n")
        sys.stdout.flush()

    def run(self) -> None:
        """Main interactive simulation loop with MuJoCo viewer."""
        print("=" * 70)
        print("  DISASTER RESPONSE DRONE SIMULATION - CONGESTED 10 KM CITY")
        print("=" * 70)
        print(f"Environment: {ENVIRONMENT_SIZE/1000:.0f} km x {ENVIRONMENT_SIZE/1000:.0f} km (Seed: {self.seed})")
        print(f"Primitives : {len(self.obstacles)} objects (Buildings, Rubble, Trees, Poles)")
        print("Controls   :")
        print("  W / S         : Move Forward / Backward (+/- Y)")
        print("  A / D         : Move Left / Right (-/+ X)")
        print("  SPACE / E     : Move Up (+Z)")
        print("  SHIFT / C / Q : Move Down (-Z)")
        print("  CTRL / F      : Sprint (5x speed boost across 10 km)")
        print("  N or P        : Spawn / Despawn Drone D02 (100m inter-drone test)")
        print("  M             : Spawn / Despawn Drone D03")
        print("  TAB           : Switch pilot control between active drones")
        print("  R             : Reset current drone to origin")
        print("  ESC           : Close simulation")
        print("=" * 70)

        # Clear screen once at startup
        os.system('cls' if os.name == 'nt' else 'clear')

        last_hud_time = 0.0
        hud_interval = 0.15  # Update HUD ~7 times/sec
        last_step_time = time.time()

        enable_high_resolution_timer()
        try:
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
                # Set initial camera reasonably close to drone so user sees immediate scene
                viewer.cam.lookat[:] = [0.0, 0.0, 5.0]
                viewer.cam.distance = 25.0
                viewer.cam.elevation = -15.0
                viewer.cam.azimuth = 90.0

                while viewer.is_running():
                    now = time.time()
                    dt = min(now - last_step_time, 0.05)
                    last_step_time = now

                    # 1. Direct manual position control
                    delta = self._poll_inputs(dt)
                    if np.any(delta != 0):
                        self.current_drone.move(delta)

                    # 2. Sync mocap positions and compute forward kinematics (ultra-fast ~0.17ms)
                    self._sync_mocap()
                    mujoco.mj_kinematics(self.model, self.data)

                    # 3. Inter-drone detection (100m radius)
                    nearby = get_nearby_drones(self.current_drone, self.drones)

                    # 4. Obstacle detection (vectorized pre-filter + exact surface distance)
                    nearest_obs, obs_dist, obs_rel = get_nearest_obstacle(
                        self.current_drone, self.obstacles, self.obstacle_positions_matrix
                    )

                    # 5. Collision detection (reuses precalculated nearest obstacle)
                    is_colliding, coll_name, prox_alert, prox_drone = check_collisions(
                        self.current_drone, self.obstacles, self.drones, min_separation=10.0,
                        obstacle_positions_matrix=self.obstacle_positions_matrix,
                        nearest_obs=nearest_obs, obs_dist=obs_dist
                    )

                    # Collision counter
                    if is_colliding and not self.was_colliding:
                        self.total_collisions += 1
                    self.was_colliding = is_colliding

                    # 6. Telemetry Display
                    if now - last_hud_time >= hud_interval:
                        self.print_hud(
                            self.current_drone,
                            nearby,
                            nearest_obs,
                            obs_dist,
                            obs_rel,
                            is_colliding,
                            coll_name,
                            prox_alert,
                            prox_drone,
                        )
                        last_hud_time = now

                    # 7. Sync MuJoCo viewer
                    viewer.sync()

                    # High-precision smooth frame pacing (maintains 60 Hz without timer jitter)
                    elapsed = time.time() - now
                    sleep_time = 0.01666 - elapsed
                    if sleep_time > 0.001:
                        time.sleep(sleep_time)
        finally:
            disable_high_resolution_timer()

        print("\n[Shutdown] Simulation ended cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Congested 10km Disaster City Drone Simulation")
    parser.add_argument("--seed", type=int, default=ENVIRONMENT_SEED, help="Random seed")
    parser.add_argument("--buildings", type=int, default=NUM_BUILDINGS, help="Number of buildings")
    parser.add_argument("--rubble_zones", type=int, default=NUM_RUBBLE_ZONES, help="Number of rubble zones")
    parser.add_argument("--rubble_per_zone", type=int, default=NUM_RUBBLE_PER_ZONE, help="Pieces per rubble zone")
    parser.add_argument("--trees", type=int, default=NUM_TREES, help="Number of trees")
    parser.add_argument("--poles", type=int, default=NUM_POLES, help="Number of poles")
    args = parser.parse_args()

    sim = DroneSimulator(
        seed=args.seed,
        num_buildings=args.buildings,
        num_rubble_zones=args.rubble_zones,
        num_rubble_per_zone=args.rubble_per_zone,
        num_trees=args.trees,
        num_poles=args.poles,
    )
    sim.run()
