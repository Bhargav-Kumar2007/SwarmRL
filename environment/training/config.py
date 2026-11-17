"""Shared configuration + path bootstrap. No torch import here (baselines and
benchmark must run with NumPy only)."""
import sys
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]


def add_repo_paths():
    for p in (str(ROOT), str(ROOT / "environment"), str(ROOT / "training")):
        if p not in sys.path:
            sys.path.insert(0, p)


CURRICULUM_STAGES = (2, 5, 10, 25, 50)
# Default per-stage budgets in AGENT transitions. These are starting points,
# not validated claims -- promote only after evaluate.py shows progress.
DEFAULT_STAGE_STEPS = {2: 300_000, 5: 400_000, 10: 600_000, 25: 900_000, 50: 1_500_000}


@dataclass
class MAPPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_clip: float = 0.2
    ppo_epochs: int = 5
    num_minibatches: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 10.0
    actor_hidden: int = 64
    critic_hidden: int = 128