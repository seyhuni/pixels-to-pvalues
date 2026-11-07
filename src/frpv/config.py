"""Configuration handling: YAML base config + experiment overrides + CLI dot-overrides.

Usage
-----
cfg = load_config("configs/base.yaml", "configs/exp/main.yaml", overrides=["train.epochs=5"])
cfg.model.encoder  -> attribute access
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return v

    def __setattr__(self, k, v):
        self[k] = v

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self))


def _wrap(d: Any) -> Any:
    if isinstance(d, dict):
        return Cfg({k: _wrap(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_wrap(v) for v in d]
    return d


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(s: str) -> Any:
    try:
        v = yaml.safe_load(s)
    except Exception:
        return s
    if isinstance(v, str):          # YAML 1.1 reads "3e-4" as a string
        try:
            return float(v)
        except ValueError:
            return v
    return v


def apply_overrides(d: dict, overrides: Iterable[str]) -> dict:
    d = copy.deepcopy(d)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got {ov}")
        key, val = ov.split("=", 1)
        cur = d
        parts = key.strip().split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = _parse_value(val)
    return d


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(base: str | Path, *exp_files: str | Path, overrides: Iterable[str] = (),
                extra: dict | None = None) -> Cfg:
    d = load_yaml(resolve_path(base))
    for f in exp_files:
        if f:
            d = deep_merge(d, load_yaml(resolve_path(f)))
    if extra:
        d = deep_merge(d, extra)
    d = apply_overrides(d, overrides)
    return _wrap(d)


def resolve_path(p: str | Path) -> Path:
    """Relative paths are resolved against the repository root (not the CWD)."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(json.loads(json.dumps(cfg)), f, sort_keys=False, allow_unicode=True)
