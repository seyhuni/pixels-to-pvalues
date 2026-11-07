"""Per-image metrics (all computed inside the ROI, as recommended by the dataset authors)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


def iou(a, b) -> float:
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def _surface(m):
    return m & ~ndimage.binary_erosion(m, border_value=0)


def msd(a, b, spacing=1.0) -> float:
    """Mean symmetric surface distance in pixels (nan if one mask is empty)."""
    if a.sum() == 0 or b.sum() == 0:
        return float("nan") if (a.sum() + b.sum()) > 0 else 0.0
    sa, sb = _surface(a), _surface(b)
    da = ndimage.distance_transform_edt(~sb, sampling=spacing)[sa]
    db = ndimage.distance_transform_edt(~sa, sampling=spacing)[sb]
    return float((da.sum() + db.sum()) / (len(da) + len(db)))


def fi(wall: np.ndarray, fib: np.ndarray) -> float:
    w = wall.sum()
    return float(fib.sum() / w) if w > 0 else float("nan")


def decode(p_wall: np.ndarray, p_fib: np.ndarray, head: str, roi: np.ndarray, t_wall=0.5, t_fib=0.5):
    """Hierarchical decoding: wall = p_wall>t; fibrosis = wall & p(fib|wall)>t (nested/flat).
    'multilabel' decodes the two maps independently (containment may be violated)."""
    wall = (p_wall > t_wall) & roi
    if head == "multilabel":
        fib = (p_fib > t_fib) & roi
    else:
        fib = wall & (p_fib > t_fib * p_wall)
    return wall, fib


def ged(preds: list, gts: list) -> float:
    """Generalised energy distance with d = 1 - IoU (Kohl et al. 2018)."""
    d = lambda a, b: 1.0 - iou(a, b)
    cross = np.mean([d(p, g) for p in preds for g in gts])
    pp = np.mean([d(p, q) for p in preds for q in preds]) if len(preds) > 1 else 0.0
    gg = np.mean([d(g, h) for g in gts for h in gts]) if len(gts) > 1 else 0.0
    return float(2 * cross - pp - gg)


def image_metrics(wall, fib, experts: dict, prefix="", surface=True) -> dict:
    """experts: {'e1': (wall, fib), 'e2': (wall, fib)} (already ROI-restricted)."""
    out = {f"{prefix}wall_area": float(wall.sum()), f"{prefix}fib_area": float(fib.sum()),
           f"{prefix}FI_hat": fi(wall, fib),
           f"{prefix}containment_viol": float((fib & ~wall).sum())}
    for k, (ew, ef) in experts.items():
        out[f"{prefix}dice_wall_{k}"] = dice(wall, ew)
        out[f"{prefix}dice_fib_{k}"] = dice(fib, ef)
        out[f"{prefix}iou_fib_{k}"] = iou(fib, ef)
        if surface:
            out[f"{prefix}msd_wall_{k}"] = msd(wall, ew)
            out[f"{prefix}msd_fib_{k}"] = msd(fib, ef)
    return out
