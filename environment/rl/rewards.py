"""Modular, configurable reward calculator with per-component tracking.

All terms use sensor/mission events only; no term references the position of an
undiscovered victim. A configurable fraction of the team's positive
discovery/exploration reward is broadcast to everyone (MAPPO shared reward).
"""
from dataclasses import dataclass, fields


@dataclass
class RewardWeights:
    discover: float = 2.0            # team-first faint discovery of a victim
    confirm: float = 6.0             # team-first confirmation (< 10 m, LOS)
    coop_confirm: float = 1.5        # confirming a victim a teammate found first
    personal_faint: float = 0.1      # own first faint contact (already team-known)
    new_cell: float = 0.08           # team-first visit of a 25 m grid cell
    personal_new_cell: float = 0.02  # own first visit of a team-known cell
    revisit: float = -0.01           # re-entering a team-explored cell
    collision_building: float = -1.0   # per colliding step
    collision_rubble: float = -0.5     # per colliding step
    collision_tree: float = -0.5       # per colliding step
    collision_pole: float = -0.5       # per colliding step
    collision_terrain: float = -1.0    # per colliding step
    separation: float = -0.15        # per step closer than min_separation
    safe_step: float = 0.002         # clean step (no collision, no proximity)
    fire_proximity: float = -0.15    # scaled by (1 - dist/25) per fire alert
    idle: float = -0.02              # moved < 20% of max step with no events
    time_cost: float = -0.005        # per-step mission efficiency cost
    team_weight: float = 0.5         # broadcast fraction of team event reward

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in known})


_TEAM_KEYS = ("discover", "confirm", "coop", "new_cell")


class RewardCalculator:
    def __init__(self, weights=None):
        self.w = weights if weights is not None else RewardWeights()

    def individual(self, events, *, colliding, collision_type, prox_alert,
                   fire_alerts, moved_frac, team_new_cell, self_new_cell,
                   cell_changed):
        """Return (individual_reward, component_dict) for one drone this step."""
        w = self.w
        c = {}
        c["discover"] = w.discover * events["team_discoveries"]
        c["confirm"] = w.confirm * events["team_confirmations"]
        c["coop"] = w.coop_confirm * events["coop_confirmations"]
        c["personal_faint"] = w.personal_faint * events["personal_new_faint"]
        c["new_cell"] = (w.new_cell * float(team_new_cell)
                         + w.personal_new_cell * float(self_new_cell and not team_new_cell))
        c["revisit"] = w.revisit if (cell_changed and not team_new_cell) else 0.0
        if colliding:
            key = "collision_" + (collision_type or "building").lower()
            c["collision"] = getattr(w, key, w.collision_building)
        else:
            c["collision"] = 0.0
        c["separation"] = w.separation if prox_alert else 0.0
        c["safe"] = w.safe_step if (not colliding and not prox_alert) else 0.0
        c["fire"] = (w.fire_proximity * sum(max(0.0, 1.0 - a["distance"] / 25.0)
                                            for a in fire_alerts)) if fire_alerts else 0.0
        useful = (events["team_discoveries"] or events["team_confirmations"]
                  or events["personal_new_faint"] or team_new_cell or self_new_cell)
        c["idle"] = w.idle if (moved_frac < 0.2 and not useful) else 0.0
        c["time"] = w.time_cost
        return float(sum(c.values())), c

    def combine(self, individual_rewards, component_dicts):
        """Add the shared team fraction; return (totals, team_pool, bonus)."""
        team_pool = 0.0
        if component_dicts:
            team_pool = float(sum(sum(cd[k] for k in _TEAM_KEYS)
                                  for cd in component_dicts) / len(component_dicts))
        bonus = self.w.team_weight * team_pool
        return [r + bonus for r in individual_rewards], team_pool, bonus