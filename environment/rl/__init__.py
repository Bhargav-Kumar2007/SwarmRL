"""Lightweight multi-agent RL layer for SwarmRL (additive; sim untouched)."""
import sys as _sys
from pathlib import Path as _Path
_ENV_DIR = str(_Path(__file__).resolve().parents[1])
if _ENV_DIR not in _sys.path:
    _sys.path.insert(0, _ENV_DIR)