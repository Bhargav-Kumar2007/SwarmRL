"""
Automated Test Suite for Congested 10 km x 10 km MuJoCo Drone Prototype.
Validates 10km procedural generation, enlarged structures, heavy disaster rubble,
seed reproducibility, safe starting area, typed obstacle sensing, and collision monitoring.
"""

import sys
import os
import numpy as np
import mujoco

from environment_generator import ProceduralEnvironmentGenerator
from drone_model import (
    Drone,
    Obstacle,
    get_nearby_drones,
    get_nearest_obstacle,
    check_collisions,
)


def run_tests():
    print("==================================================")
    print(" RUNNING CONGESTED 10 KM CITY TEST SUITE")
    print("==================================================")

    # 1. Determinism & Seed Reproducibility
    print("[Test 1] Testing Deterministic Procedural Generation (10 km x 10 km)...")
    gen1 = ProceduralEnvironmentGenerator(env_size=10000.0, seed=12345)
    xml1, obs1 = gen1.generate()

    gen2 = ProceduralEnvironmentGenerator(env_size=10000.0, seed=12345)
    xml2, obs2 = gen2.generate()

    gen3 = ProceduralEnvironmentGenerator(env_size=10000.0, seed=99999)
    xml3, obs3 = gen3.generate()

    assert xml1 == xml2, "Generators with identical seed (12345) must produce identical XML output!"
    assert len(obs1) == len(obs2), "Obstacle lists must be identical with same seed!"
    assert np.allclose(obs1[0].pos, obs2[0].pos), "First obstacle position must match!"
    assert xml1 != xml3, "Generators with different seeds must produce different output!"
    print(f"  Seed reproducibility verified: Seed 12345 produces exactly {len(obs1)} identical objects.")

    # 2. Safe Takeoff Plaza Check (200m x 200m box around origin)
    print("[Test 2] Testing Safe Takeoff Plaza around origin (0, 0)...")
    safe_margin = 100.0  # 200m x 200m box centered at (0, 0)
    for obs in obs1:
        in_safe_box = abs(obs.pos[0]) < safe_margin and abs(obs.pos[1]) < safe_margin
        assert not in_safe_box, f"Obstacle {obs.name} generated inside safe starting zone at ({obs.pos[0]:.1f}, {obs.pos[1]:.1f})!"
    print(f"  Safe takeoff plaza verified: 0 of {len(obs1)} obstacles violate 200m x 200m origin box.")

    # 3. Environment Scale and Congested Object Counts
    print("[Test 3] Testing 10 km Bounds and Structure Counts...")
    bldgs = [o for o in obs1 if o.obs_type == "Building"]
    rubble = [o for o in obs1 if o.obs_type == "Rubble"]
    trees = [o for o in obs1 if o.obs_type == "Tree"]
    poles = [o for o in obs1 if o.obs_type == "Pole"]

    print(f"  Counts: {len(bldgs)} Buildings, {len(rubble)} Rubble, {len(trees)} Trees, {len(poles)} Poles (Total: {len(obs1)})")
    assert len(bldgs) >= 800, f"Expected at least 800 buildings, got {len(bldgs)}"
    assert len(rubble) >= 1400, f"Expected at least 1400 rubble pieces, got {len(rubble)}"
    assert len(trees) >= 850, f"Expected at least 850 trees, got {len(trees)}"
    assert len(poles) >= 300, f"Expected at least 300 poles, got {len(poles)}"

    # All objects within 10 km bounds (-5000 to +5000 m)
    for obs in obs1:
        assert abs(obs.pos[0]) <= 5000.0, f"Obstacle {obs.name} exceeds X boundary: {obs.pos[0]}"
        assert abs(obs.pos[1]) <= 5000.0, f"Obstacle {obs.name} exceeds Y boundary: {obs.pos[1]}"
    print("  Structure counts and 10 km bounds (-5000m to +5000m) verified.")

    # 4. Check Enlarged Dimensions
    print("[Test 4] Verifying Enlarged Structure Dimensions...")
    max_bldg_h = max(b.size_params[2] for b in bldgs)
    max_rubble_size = max(max(r.size_params) for r in rubble)
    max_tree_crown = max(t.size_params[2] for t in trees)
    max_pole_h = max(p.size_params[1] for p in poles)

    print(f"  Max Building Height: {max_bldg_h:.1f} m (Towers up to 250m)")
    print(f"  Max Rubble Dimension: {max_rubble_size:.1f} m (Slabs/Pillars up to 15m)")
    print(f"  Max Tree Crown Radius: {max_tree_crown:.1f} m")
    print(f"  Max Pole/Tower Height: {max_pole_h:.1f} m (Masts up to 70m)")

    assert max_bldg_h >= 150.0, f"Expected towers exceeding 150m, got {max_bldg_h}"
    assert max_rubble_size >= 10.0, f"Expected large rubble pieces >= 10m, got {max_rubble_size}"
    assert max_tree_crown >= 10.0, f"Expected large tree crowns >= 10m, got {max_tree_crown}"
    assert max_pole_h >= 45.0, f"Expected tall telecom/power towers >= 45m, got {max_pole_h}"
    print("  Enlarged dimensions verified across all categories.")

    # 5. MuJoCo Model Compilation of Generated Scene
    print("[Test 5] Compiling generated_scene.xml with MuJoCo...")
    gen1.export_to_file("generated_scene.xml")
    model = mujoco.MjModel.from_xml_path("generated_scene.xml")
    data = mujoco.MjData(model)
    print(f"  Compiled successfully! Geoms: {model.ngeom}, Bodies: {model.nbody}, Mocaps: {model.nmocap}")
    assert model.nmocap >= 3, f"Expected at least 3 mocap bodies, got {model.nmocap}"

    data.mocap_pos[0] = [0.0, 0.0, 5.0]
    mujoco.mj_forward(model, data)
    print(f"  Kinematics step completed with 0 errors on {model.ngeom} geoms.")

    # 6. Drone Boundary Clamping in 10 km World
    print("[Test 6] Testing Drone Boundary Clamping (-5,000m to +5,000m)...")
    drone = Drone(id="D01", pos=np.array([0.0, 0.0, 5.0]), x_bounds=(-5000.0, 5000.0), y_bounds=(-5000.0, 5000.0))
    drone.move(np.array([10000.0, 0.0, 0.0]))  # Attempt to fly past +5,000m
    assert drone.pos[0] == 5000.0, f"Expected X clamped to 5000.0, got {drone.pos[0]}"
    drone.move(np.array([-20000.0, 0.0, 0.0]))  # Attempt past -5,000m
    assert drone.pos[0] == -5000.0, f"Expected X clamped to -5000.0, got {drone.pos[0]}"
    drone.set_position(0.0, 0.0, -50.0)  # Attempt below ground
    assert drone.pos[2] == 0.2, f"Expected Z clamped to 0.2, got {drone.pos[2]}"
    print("  Drone 10 km boundary clamping verified.")

    # 7. Typed Nearest Obstacle Detection
    print("[Test 7] Testing Typed Obstacle Detection & Primitive Surface Distances...")
    runtime_obs = [
        Obstacle(name=o.name, obs_type=o.obs_type, pos=o.pos, shape=o.shape, size_params=o.size_params)
        for o in obs1
    ]
    drone.set_position(0.0, 0.0, 10.0)
    nearest, dist, rel = get_nearest_obstacle(drone, runtime_obs)
    assert nearest is not None
    assert dist > 0.0
    print(f"  Nearest obstacle from takeoff plaza: {nearest.name} ({nearest.obs_type}) at {dist:.1f} m (Rel: ({rel[0]:.1f}, {rel[1]:.1f}, {rel[2]:.1f}))")

    # 8. Collision and Proximity Alerts
    print("[Test 8] Testing Collision and Proximity alerts...")
    bldg_test = Obstacle(name="Building_Test", obs_type="Building", pos=np.array([100.0, 0.0, 20.0]), shape="box", size_params=np.array([40.0, 40.0, 40.0]))
    d1 = Drone(id="D01", pos=np.array([0.0, 0.0, 10.0]), comm_radius=100.0)
    d2 = Drone(id="D02", pos=np.array([42.0, -51.0, 8.0]), active=True)

    is_coll, coll_name, prox_alert, prox_id = check_collisions(d1, [bldg_test], [d1, d2])
    assert not is_coll and not prox_alert, "Expected no collision at safe position"

    # Terrain collision (Z <= collision_radius)
    d1.set_position(0.0, 0.0, 1.0)
    is_coll, coll_name, _, _ = check_collisions(d1, [bldg_test], [d1, d2])
    assert is_coll and coll_name == "Terrain", f"Expected Terrain collision, got {is_coll}, {coll_name}"

    # Building collision
    d1.set_position(100.0, 0.0, 20.0)
    is_coll, coll_name, _, _ = check_collisions(d1, [bldg_test], [d1, d2])
    assert is_coll and coll_name == "Building_Test", f"Expected building collision, got {is_coll}, {coll_name}"

    # Drone-to-drone separation (<10m)
    d1.set_position(42.0, -51.0, 10.0)
    is_coll, _, prox_alert, prox_id = check_collisions(d1, [bldg_test], [d1, d2], min_separation=10.0)
    assert prox_alert and prox_id == "D02", f"Expected proximity alert with D02, got {prox_alert}, {prox_id}"
    print("  Collisions (Terrain, Building) and Inter-Drone Separation (<10m) verified.")

    print("==================================================")
    print(" ALL 8 TEST SUITES PASSED PERFECTLY!")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
