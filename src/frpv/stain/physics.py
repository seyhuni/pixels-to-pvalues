"""Beer-Lambert stain physics for the three-dye Picro-Mallory (Mallory trichrome family) stain.

Optical density  OD = -log((I + 1) / 256)  (per RGB channel, I in [0,255]).
Beer-Lambert:    OD = C @ M,   C (N x 3) non-negative dye concentrations,
                 M (3 x 3) rows = unit OD vectors of the dyes.
Dyes (row order used everywhere): 0 aniline blue (collagen), 1 acid fuchsin (muscle/cytoplasm,
nuclei), 2 orange G (erythrocytes, picric acid component).

This module provides
  * estimate_stain_matrix(): data-driven 3-dye estimation by NMF on OD pixels, matched to dye
    reference directions with the Hungarian algorithm;
  * BeerLambertAugment: physically consistent stain augmentation (jitter concentrations and
    rotate dye vectors, then re-synthesise RGB) - a trichrome analogue of HED augmentation;
  * StainDeconv (torch): differentiable, optionally learnable deconvolution layer whose output
    concentration maps feed the high-resolution detail branch of the network.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DYES = ["aniline_blue", "acid_fuchsin", "orange_g"]

# Approximate OD directions (R, G, B) used only to *name/order* NMF components and as fallback.
REFERENCE_VECTORS = np.array([
    [0.80, 0.55, 0.25],   # aniline blue  -> absorbs red (appears blue)
    [0.20, 0.85, 0.48],   # acid fuchsin  -> absorbs green (appears magenta/red)
    [0.05, 0.35, 0.94],   # orange G      -> absorbs blue (appears orange/yellow)
], dtype=np.float64)
REFERENCE_VECTORS /= np.linalg.norm(REFERENCE_VECTORS, axis=1, keepdims=True)


def rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    return -np.log((rgb.astype(np.float32) + 1.0) / 256.0)


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    return np.clip(256.0 * np.exp(-od) - 1.0, 0, 255)


def normalize_rows(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-8)


def match_to_reference(vectors: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment
    v = normalize_rows(vectors)
    cost = -v @ REFERENCE_VECTORS.T          # rows: estimated, cols: reference
    r, c = linear_sum_assignment(cost)
    out = np.zeros_like(v)
    out[c] = v[r]
    return out


def estimate_stain_matrix(od_pixels: np.ndarray, method: str = "vertex", q: float = 0.99,
                          iters: int = 3, seed: int = 0) -> np.ndarray:
    """Estimate the 3 dye OD vectors from tissue pixels (N,3) - no labels used.

    vertex (default): reference-guided pure-pixel estimation. Each dye vector is the mean OD
        chromaticity of the q-quantile of pixels most aligned with the current dye direction;
        iterated `iters` times starting from REFERENCE_VECTORS (a 3-stain generalisation of
        Macenko's extreme-angle idea).
    nmf: non-negative matrix factorisation (tends to put vectors on the orthant boundary).
    """
    od = np.clip(od_pixels, 0, None)
    if method == "nmf":
        from sklearn.decomposition import NMF
        nmf = NMF(n_components=3, init="nndsvda", max_iter=800, random_state=seed)
        nmf.fit(od)
        return match_to_reference(nmf.components_)
    u = od / np.maximum(np.linalg.norm(od, axis=1, keepdims=True), 1e-8)
    v = REFERENCE_VECTORS.copy()
    for _ in range(iters):
        new = []
        for r in v:
            c = u @ r
            new.append(u[c >= np.quantile(c, q)].mean(0))
        v = normalize_rows(np.array(new))
    return v


def estimate_from_cache(stems, cache_dir, n_px=2000, max_images=300, seed=0):
    """Stain matrix + concentration stats from a set of cached images (e.g. the TRAINING fold only,
    so that no colour information from held-out animals/projects is used)."""
    from pathlib import Path as _P
    rng = np.random.default_rng(seed)
    stems = list(stems)
    if len(stems) > max_images:
        stems = list(rng.choice(stems, size=max_images, replace=False))
    samples = []
    for st in stems:
        img = np.load(_P(cache_dir) / "img" / f"{st}.npy", mmap_mode="r")
        lab = np.load(_P(cache_dir) / "lab" / f"{st}.npy", mmap_mode="r")
        roi = (np.asarray(lab) & 1) > 0
        od = rgb_to_od(np.asarray(img)[roi])
        od = od[od.sum(1) > 0.3]
        if len(od):
            samples.append(od[rng.choice(len(od), size=min(n_px, len(od)), replace=False)])
    x = np.concatenate(samples)
    m = estimate_stain_matrix(x)
    c = x @ np.linalg.pinv(m)
    return m, {"mean": c.mean(0).round(5).tolist(), "std": c.std(0).round(5).tolist()}


def save_stain_vectors(path: Path, global_m: np.ndarray, per_project: dict, stats: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"dyes": DYES, "global": global_m.tolist(),
               "per_project": {k: v.tolist() for k, v in per_project.items()},
               "concentration_stats": stats}
    Path(path).write_text(json.dumps(payload, indent=2))


def load_stain_matrix(path: Path | None) -> np.ndarray:
    if path is not None and Path(path).exists():
        return np.array(json.loads(Path(path).read_text())["global"], dtype=np.float64)
    return REFERENCE_VECTORS.copy()


def deconvolve(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    """rgb (H,W,3) uint8 -> concentrations (H,W,3)."""
    od = rgb_to_od(rgb).reshape(-1, 3)
    c = od @ np.linalg.pinv(m)
    return c.reshape(rgb.shape[:2] + (3,))


def _rotate_towards_random(v: np.ndarray, deg: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate unit vector v by `deg` degrees towards a random orthogonal direction; keep OD >= 0."""
    if deg <= 0:
        return v
    r = rng.normal(size=3)
    r -= r.dot(v) * v
    r /= np.linalg.norm(r) + 1e-8
    a = np.deg2rad(rng.uniform(-deg, deg))
    w = np.cos(a) * v + np.sin(a) * r
    w = np.clip(w, 1e-3, None)
    return w / np.linalg.norm(w)


class BeerLambertAugment:
    """Physically consistent stain augmentation in dye-concentration space."""

    def __init__(self, m: np.ndarray, conc_scale=0.2, conc_shift=0.04, vector_deg=4.0):
        self.m = normalize_rows(np.asarray(m, np.float64))
        self.m_pinv = np.linalg.pinv(self.m)
        self.conc_scale, self.conc_shift, self.vector_deg = conc_scale, conc_shift, vector_deg

    def __call__(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w, _ = rgb.shape
        od = rgb_to_od(rgb).reshape(-1, 3)
        c = od @ self.m_pinv
        a = rng.uniform(1 - self.conc_scale, 1 + self.conc_scale, size=3)
        b = rng.uniform(-self.conc_shift, self.conc_shift, size=3)
        c = c * a + b
        m2 = np.stack([_rotate_towards_random(v, self.vector_deg, rng) for v in self.m])
        od2 = np.clip(c @ m2, 0, None)
        return np.rint(od_to_rgb(od2)).reshape(h, w, 3).astype(np.uint8)


def hsv_augment(rgb: np.ndarray, rng: np.random.Generator, h=0.04, s=0.15, v=0.1) -> np.ndarray:
    """Generic colour jitter (non-physical ablation baseline)."""
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-h, h) * 180) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1 - s, 1 + s), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(1 - v, 1 + v), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ----------------------------------------------------------------------------- torch layer
try:
    import torch
    import torch.nn as nn

    class StainDeconv(nn.Module):
        """Differentiable Beer-Lambert deconvolution.

        input : RGB in [0,1], (B,3,H,W)
        output: standardised dye concentrations (B,3,H,W)
        The stain matrix is M = normalize(M0 + dM); dM is learnable (if learnable) and
        penalised by `regularizer()` so the layer stays physically interpretable.
        """

        def __init__(self, m0: np.ndarray, learnable: bool = True,
                     conc_mean=None, conc_std=None):
            super().__init__()
            self.register_buffer("m0", torch.tensor(normalize_rows(m0), dtype=torch.float32))
            self.dm = nn.Parameter(torch.zeros(3, 3), requires_grad=learnable)
            cm = torch.tensor(conc_mean if conc_mean is not None else [0.3, 0.3, 0.2], dtype=torch.float32)
            cs = torch.tensor(conc_std if conc_std is not None else [0.3, 0.3, 0.2], dtype=torch.float32)
            self.register_buffer("c_mean", cm.view(1, 3, 1, 1))
            self.register_buffer("c_std", cs.view(1, 3, 1, 1).clamp_min(1e-3))

        def matrix(self) -> torch.Tensor:
            m = (self.m0 + self.dm).clamp_min(1e-4)
            return m / m.norm(dim=1, keepdim=True)

        def forward(self, rgb01: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type=rgb01.device.type, enabled=False):
                x = rgb01.float()
                od = -torch.log((x * 255.0 + 1.0) / 256.0)                # (B,3,H,W)
                minv = torch.linalg.pinv(self.matrix())                    # (3,3)
                c = torch.einsum("bchw,cd->bdhw", od, minv)
                return (c - self.c_mean) / self.c_std

        def regularizer(self) -> torch.Tensor:
            return (self.dm ** 2).sum()
except ImportError:  # pragma: no cover
    pass
