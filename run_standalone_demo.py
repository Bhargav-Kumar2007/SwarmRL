"""
Convenience launcher for the Congested 10 km MuJoCo 3D Drone Prototype.
Runs the simulation entrypoint in main.py, supporting optional seed and density flags.
"""

import sys
import argparse
from main import (
    DroneSimulator,
    ENVIRONMENT_SEED,
    NUM_BUILDINGS,
    NUM_RUBBLE_ZONES,
    NUM_RUBBLE_PER_ZONE,
    NUM_TREES,
    NUM_POLES,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Congested 10km Disaster City Drone Simulation")
    parser.add_argument("--seed", type=int, default=ENVIRONMENT_SEED, help="Random seed (deterministic)")
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
