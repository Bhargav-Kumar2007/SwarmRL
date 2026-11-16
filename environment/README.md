# 3D Disaster-Response Drone Prototype with Congested 10 km × 10 km Environment

An interactive 3D drone prototype built using **MuJoCo** and **Python**, featuring an **ultra-congested, enlarged 10 km × 10 km procedural disaster environment generator** as the foundation for a multi-agent disaster-response swarm project.

---

## What's New: 10 km × 10 km Ultra-Congested Disaster Environment

1. **Compact 10 km × 10 km Footprint ($100\text{ km}^2$)**:
   - 4 times lighter in spatial area than the previous 20 km world ($100\text{ km}^2$ vs $400\text{ km}^2$).
   - Spans from **$X \in [-5000, +5000]$ m** and **$Y \in [-5000, +5000]$ m**.
   - Faster spatial density, tight visual horizons, and highly immersive flight corridors.

2. **Massive Density Increase (3,545 Objects Total)**:
   - **800 Buildings**:
     - Dense urban blocks with narrow alleys (**10–24 m gaps**) and towering street canyons.
     - **Mega Skyscrapers**: Height up to **250 m** (widths 45–80 m).
     - **Commercial High-Rises**: Height 60–120 m.
     - **Mid-Rise Apartments**: Height 30–65 m.
     - **Massive Warehouses**: Footprints up to 130 m wide.
     - **Damaged / Fractured Structures**: Broken buildings with rubble spilling into streets.
   - **1,500 Rubble Pieces (50 Clusters × 30 Pieces)**:
     - **Disaster Ground Zero**: An intensive catastrophe sector located right next to the takeoff plaza, packed with dense rubble debris.
     - **Massive Tilted Slabs**: Giant collapsed concrete slabs up to **12 m** wide with randomized pitch/roll/yaw.
     - **Fallen Pillars & Pipes**: Heavy industrial conduits up to **15 m** long and **3.5 m** radius.
     - **Giant Debris Mounds**: Boulders and debris heaps up to **7 m** radius.
   - **900 Trees**:
     - Urban avenue trees along street corridors.
     - Large city parks and dense forest patches.
     - Giant foliage: Trunk heights up to **24 m** and crown foliage radii up to **16 m**!
   - **350 Poles & Transmission Towers**:
     - Street lighting (8–16 m), utility power poles (16–28 m), and massive telecom/broadcast towers up to **70 m** tall!

3. **Deterministic Seeding (Manually Changeable)**:
   - Easily changed via the `ENVIRONMENT_SEED` configuration variable or `--seed <int>` CLI argument.
   - Generates identical environments with the same seed, and completely unique cities with different seeds.

4. **Safe Takeoff Plaza**:
   - The 200 m × 200 m square around $(0, 0)$ is strictly preserved as an open clearing, with dense urban skyscrapers, trees, and disaster rubble starting immediately at 100 m distance!

---

## Configuration Variables

At the top of [`main.py`](file:///d:/PROJECTS/50Drone_RL_System/main.py):

```python
ENVIRONMENT_SIZE = 10000.0       # 10 km x 10 km square (-5,000m to +5,000m)
ENVIRONMENT_SEED = 12345         # Deterministic random seed (change manually or via --seed)

# Congested Density Counts
NUM_BUILDINGS = 800              # Skyscrapers, high-rises, mid-rises, dense row-houses
NUM_RUBBLE_ZONES = 50            # Disaster rubble clusters (heavy concentration in disaster zone)
NUM_RUBBLE_PER_ZONE = 30         # Pieces per cluster (Total: 1,500 rubble pieces)
NUM_TREES = 900                  # Urban parks, street avenues, forest patches
NUM_POLES = 350                  # Streetlights, utility poles, telecom towers
AMBIENT_TEMP_C = 18.0            # Baseline air/surface temperature
NUM_FIRE_ZONES = 30              # Each zone contains 1–4 fire sources
NUM_COLD_ZONES = 12              # Park, forest, and shaded cold pockets
COLD_ZONE_DELTA_C = (-10.0, -4.0)
NUM_HUMANS = 150                 # Victims, including two plaza demo targets
FAINT_SIGNAL_RADIUS = 50.0       # Warm blob contact radius (m)
CONFIRM_RADIUS = 10.0            # Classified human radius (m)
```

## Thermal Search Model

The seeded map includes ground, indoor, and rooftop fires, cold park pockets, and static human victims. A drone receives a faint human contact within **50 m** and confirms the victim within **10 m**, subject to line of sight. Indoor victims and fires are detectable only when the drone is over the parent building footprint (expanded by 2 m). Strong local fire heat can mask a human's faint signal. Fires appear as non-colliding visual geoms and produce alerts within 25 m. Two demo victims are placed at `(30, 0, 0.9)` and `(8, 5, 0.9)`.

---

## Running the Simulation

### Default Run (Seed: 12345)
```powershell
.\.venv\Scripts\python.exe run_standalone_demo.py
```
or:
```powershell
.\.venv\Scripts\python.exe main.py
```

### Run with a Custom Random Seed
```powershell
.\.venv\Scripts\python.exe run_standalone_demo.py --seed 42
```
or:
```powershell
.\.venv\Scripts\python.exe main.py --seed 99999
```

### Run Automated Verification Suite
```powershell
.\.venv\Scripts\python.exe test_prototype.py
```

### Inspect Single Drone Model in MuJoCo
```powershell
.\.venv\Scripts\python.exe view_drone.py
```
*(Options: `--model detailed` [default] or `--model prototype`, `--no-orbit`, `--no-spin`)*

---

## Flight Controls Reference

| Key | Action |
|---|---|
| **`W`** / **`S`** | Move Forward / Backward (`+Y` / `-Y`) |
| **`A`** / **`D`** | Move Left / Right (`-X` / `+X`) |
| **`SPACE`** or **`E`** | Move Up (`+Z`) |
| **`SHIFT`**, **`C`**, or **`Q`** | Move Down (`-Z`) |
| **`CTRL`** or **`F`** | **Sprint Mode** (5x speed boost for navigating across 10 km) |
| **`N`** or **`P`** | Spawn / Despawn Drone `D02` (tests 100m inter-drone communication) |
| **`M`** | Spawn / Despawn Drone `D03` |
| **`TAB`** | Switch pilot control between active drones |
| **`R`** | Reset current drone to starting coordinates |
| **`ESC`** | Quit simulation |

---

## Telemetry HUD Display

```text
--------------------------------
Drone: D01
Environment: Congested City 10 km x 10 km (Seed: 12345)
Active Objects: 3545 (Buildings, Rubble, Trees, Poles)

Position:
X: 120.0 m
Y: 85.0 m
Z: 40.0 m

Speed:
10.0 m/s

Communication radius:
100 m

Nearby drones:
1
  D02 | Dist: 66.1 m | Rel: (42.0, -51.0, 3.0)

Nearest obstacle:
Building_024
Distance:
142.3 m
Relative position:
(82.0, -112.0, -10.0)

Collision:
NO

Total collisions:
0

Drone Separation (<10m): OK (> 10m)
--------------------------------
Flight : W/S (+/-Y), A/D (+/-X), SPACE (+Z), SHIFT/C/Q (-Z)
Sprint : Hold CTRL or F for 5x Speed Boost across 10 km map
Actions: N/P (Spawn D02), M (Spawn D03), TAB (Switch Pilot), R (Reset), ESC (Quit)
```
