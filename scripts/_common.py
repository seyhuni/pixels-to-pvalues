"""Shared helpers for the command-line scripts (adds src/ to sys.path)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frpv.config import load_config, resolve_path  # noqa: E402,F401


def results_dir(cfg=None) -> Path:
    if cfg is None:
        cfg = load_config("configs/base.yaml")
    return resolve_path(cfg.paths.results_dir)


def load_oof(name: str, cfg=None):
    import pandas as pd
    p = results_dir(cfg) / name / "oof_predictions.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run training/collection for '{name}' first")
    return pd.read_csv(p)
