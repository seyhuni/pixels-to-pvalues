"""PyTorch datasets.

TrainCrops : random square crops (ROI-centred with prob p), scale jitter, flips/rot90,
             stain augmentation (Beer-Lambert / HSV / none).
FullImages : whole native-resolution images for sliding-window inference.

Images are returned as float tensors in [0,1] (normalisation happens inside the model so that
the stain-physics layer sees physical intensities). Labels are categorical maps
(0 bg, 1 wall-only, 2 fibrosis) for each rater plus the ROI mask.
Works with num_workers > 0 on Windows (spawn): everything is picklable and the cache is
opened lazily with np.load(mmap_mode='r').
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .meta import decode_labels, to_categorical
from ..stain.physics import BeerLambertAugment, hsv_augment


def worker_init(_worker_id):
    """DataLoader worker setup: no nested thread pools inside workers (OpenCV/torch) - avoids
    over-subscription and native crashes with many spawn workers on Windows."""
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


class _CacheMixin:
    cache_dir: Path

    def _img(self, stem: str) -> np.ndarray:
        return np.load(self.cache_dir / "img" / f"{stem}.npy", mmap_mode="r")

    def _lab(self, stem: str) -> np.ndarray:
        return np.load(self.cache_dir / "lab" / f"{stem}.npy", mmap_mode="r")


def _crop_padded(img, lab, cy, cx, cs):
    """Square crop of side cs centred near (cy,cx). If cs exceeds the image, the image is
    reflect-padded and the label is zero-padded (=> ROI 0, ignored by the loss)."""
    H, W = lab.shape
    y0 = int(np.clip(cy - cs // 2, 0, max(H - cs, 0)))
    x0 = int(np.clip(cx - cs // 2, 0, max(W - cs, 0)))
    im = np.ascontiguousarray(img[y0:y0 + cs, x0:x0 + cs])
    lb = np.ascontiguousarray(lab[y0:y0 + cs, x0:x0 + cs])
    ph, pw = cs - im.shape[0], cs - im.shape[1]
    if ph > 0 or pw > 0:
        im = np.pad(im, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        lb = np.pad(lb, ((0, ph), (0, pw)), mode="constant")
    return im, lb


class TrainCrops(Dataset, _CacheMixin):
    def __init__(self, meta: pd.DataFrame, cache_dir: Path, crop_size: int, samples_per_epoch: int,
                 scale_jitter=(0.85, 1.15), roi_center_prob=0.9, stain_aug: str = "beer_lambert",
                 stain_matrix: np.ndarray | None = None, stain_kwargs: dict | None = None,
                 sample_weighting: str = "image", seed: int = 0):
        self.meta = meta.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.crop = int(crop_size)
        self.n = int(samples_per_epoch)
        self.scale_jitter = tuple(scale_jitter)
        self.roi_p = float(roi_center_prob)
        self.stain_aug = stain_aug
        self.bl = BeerLambertAugment(stain_matrix, **(stain_kwargs or {})) if stain_aug == "beer_lambert" else None
        self.seed = seed
        self.epoch = 0
        if sample_weighting == "animal":
            cnt = self.meta.groupby("animal")["file"].transform("count").to_numpy()
            w = 1.0 / cnt
        else:
            w = np.ones(len(self.meta))
        self.p = w / w.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # idx is a GLOBAL sample counter produced by EpochSampler (epoch * n + i), so crops differ
        # between epochs even with persistent DataLoader workers (Windows spawn).
        rng = np.random.default_rng((self.seed, int(idx)))
        row = self.meta.iloc[rng.choice(len(self.meta), p=self.p)]
        img = self._img(row.stem)
        lab = self._lab(row.stem)
        H, W = lab.shape
        s = rng.uniform(*self.scale_jitter)
        cs = int(round(self.crop * s))
        if rng.random() < self.roi_p:
            # pick a random ROI pixel as crop centre (sampled on a coarse grid for speed)
            ys, xs = np.nonzero(lab[::8, ::8] & 1)
            if len(ys):
                k = rng.integers(len(ys))
                cy, cx = ys[k] * 8, xs[k] * 8
            else:
                cy, cx = rng.integers(H), rng.integers(W)
        else:
            cy, cx = rng.integers(H), rng.integers(W)
        im, lb = _crop_padded(img, lab, cy, cx, cs)
        if cs != self.crop:
            im = cv2.resize(im, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)
            lb = cv2.resize(lb, (self.crop, self.crop), interpolation=cv2.INTER_NEAREST)
        # geometric: flips + rot90 (square crop)
        k = int(rng.integers(4))
        im, lb = np.rot90(im, k), np.rot90(lb, k)
        if rng.random() < 0.5:
            im, lb = im[:, ::-1], lb[:, ::-1]
        im, lb = np.ascontiguousarray(im), np.ascontiguousarray(lb)
        # colour
        if self.stain_aug == "beer_lambert" and rng.random() < 0.8:
            im = self.bl(im, rng)
        elif self.stain_aug == "hsv" and rng.random() < 0.8:
            im = hsv_augment(im, rng)
        d = decode_labels(lb)
        y1 = to_categorical(d["w1"], d["f1"])
        y2 = to_categorical(d["w2"], d["f2"])
        return {
            "image": torch.from_numpy(im).permute(2, 0, 1).float().div_(255.0),
            "y": torch.from_numpy(np.stack([y1, y2])),          # (R,H,W) int64
            "roi": torch.from_numpy(d["roi"].astype(np.float32)),
        }


class EpochSampler(Sampler):
    """Yields epoch*n ... epoch*n+n-1; call set_epoch(e) before each epoch (main process)."""

    def __init__(self, n: int):
        self.n, self.epoch = int(n), 0

    def set_epoch(self, e: int):
        self.epoch = int(e)

    def __len__(self):
        return self.n

    def __iter__(self):
        return iter(range(self.epoch * self.n, (self.epoch + 1) * self.n))


class FullImages(Dataset, _CacheMixin):
    def __init__(self, meta: pd.DataFrame, cache_dir: Path):
        self.meta = meta.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        im = np.ascontiguousarray(self._img(row.stem))
        lb = np.ascontiguousarray(self._lab(row.stem))
        return {"image": torch.from_numpy(im).permute(2, 0, 1).float().div_(255.0),
                "lab": torch.from_numpy(lb), "stem": row.stem, "file": row.file}


def collate_full(batch):
    return batch  # list of dicts (variable image sizes)
