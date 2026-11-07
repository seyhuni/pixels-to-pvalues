"""Metadata, splits and the memory-mapped cache.

Filename convention (dataset paper): {PC}{GN}-{AID}-L{LL}_{FID}.png
A unique animal is PC+GN+AID (AID numbers are re-used across projects).

Cache layout (written once by scripts/01_prepare_data.py):
  cache/img/<stem>.npy   uint8 HxWx3 RGB
  cache/lab/<stem>.npy   uint8 HxW bit-field: bit0 ROI, bit1 E1-wall, bit2 E1-fib, bit3 E2-wall, bit4 E2-fib
  cache/meta.csv         one row per image (metadata, sizes, expert areas & FI, folds)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

FNAME_RE = re.compile(r"^([A-Z]{1,2})(\d+)-(\d+)-L(\d+)_(.+)\.png$")
PROJECTS = ["AH", "M", "RA", "RE"]

# Group semantics from Table 2 of the dataset paper.
#   role: healthy | model (untreated disease) | treated
GROUP_INFO = {
    ("AH", 0): ("PAH", "healthy"), ("AH", 1): ("PAH", "model"),
    ("AH", 2): ("PAH", "treated"), ("AH", 3): ("PAH", "treated"),
    ("M", 62): ("CTEPH", "healthy"), ("M", 61): ("CTEPH", "model"),
    ("RE", 0): ("CTEPH", "healthy"), ("RE", 1): ("CTEPH", "model"),
    ("RE", 2): ("CTEPH", "treated"), ("RE", 3): ("CTEPH", "treated"),
    ("RA", 3): ("CTEPH", "model"), ("RA", 1): ("CTEPH", "treated"),
    ("RA", 2): ("CTEPH", "treated"), ("RA", 4): ("CTEPH", "treated"),
}

BIT_ROI, BIT_W1, BIT_F1, BIT_W2, BIT_F2 = 1, 2, 4, 8, 16


def parse_filename(name: str) -> dict:
    m = FNAME_RE.match(name)
    if m is None:
        raise ValueError(f"Unexpected filename: {name}")
    pc, gn, aid, ll, fid = m.groups()
    gn = int(gn)
    disease, role = GROUP_INFO.get((pc, gn), ("unknown", "unknown"))
    return dict(file=name, stem=name[:-4], pc=pc, gn=gn, aid=aid, animal=f"{pc}{gn}-{aid}",
                group=f"{pc}{gn}", level=int(ll), fid=fid, disease=disease, role=role)


def find_data_root(start: Path) -> Path:
    """Return the folder containing raw_images/ (searches below `start`)."""
    start = Path(start)
    if (start / "raw_images").is_dir():
        return start
    for p in sorted(start.rglob("raw_images")):
        if p.is_dir() and (p.parent / "expert_1").is_dir():
            return p.parent
    raise FileNotFoundError(f"Could not find raw_images/ + expert_1/ below {start}")


def decode_labels(lab: np.ndarray) -> dict:
    return dict(roi=(lab & BIT_ROI) > 0, w1=(lab & BIT_W1) > 0, f1=(lab & BIT_F1) > 0,
                w2=(lab & BIT_W2) > 0, f2=(lab & BIT_F2) > 0)


def to_categorical(wall: np.ndarray, fib: np.ndarray) -> np.ndarray:
    """0 = background, 1 = wall without fibrosis, 2 = fibrosis (fibrosis is inside wall)."""
    y = np.zeros(wall.shape, np.int64)
    y[wall] = 1
    y[fib & wall] = 2
    return y


def assign_lopo(meta: pd.DataFrame) -> pd.Series:
    return meta["pc"].map({p: i for i, p in enumerate(PROJECTS)}).astype(int)


def make_official_folds(meta: pd.DataFrame, split_csv: Path) -> pd.Series:
    s = pd.read_csv(split_csv)
    m = dict(zip(s["filename"], s["fold"]))
    missing = [f for f in meta["file"] if f not in m]
    if missing:
        raise ValueError(f"{len(missing)} images missing from official split file, e.g. {missing[:3]}")
    return meta["file"].map(m).astype(int)


def load_meta(cache_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(cache_dir) / "meta.csv", dtype={"aid": str, "fid": str})


def fold_column(split: str) -> str:
    return {"official": "fold_official", "lopo": "fold_lopo"}[split]


def inner_val_animals(train_animals: list[str], frac: float, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    a = np.array(sorted(train_animals))
    n = max(1, int(round(frac * len(a)))) if frac > 0 else 0
    return sorted(rng.choice(a, size=n, replace=False).tolist()) if n else []
