"""Sliding-window, flip-TTA, multi-checkpoint (ensemble) inference at native resolution
and per-image evaluation -> one row per image (the 'predictions table' used by all later
statistics: conformal intervals, PPI, rater analyses)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data.meta import decode_labels
from ..models.network import wall_fib_probs, rater_wall_fib
from .metrics import decode, fi, image_metrics, ged

TTA_FLIPS = [(), (3,), (2,), (2, 3)]


def _logit01(p, eps=0.01):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _gauss(w, device):
    s = w / 8.0
    ax = torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * s * s))
    m = g[:, None] * g[None, :]
    return (m / m.max()).clamp_min(1e-3)


def _starts(L, w, stride):
    if L <= w:
        return [0]
    s = list(range(0, L - w, stride))
    s.append(L - w)
    return sorted(set(s))


@torch.no_grad()
def sliding_window(model, img, window=768, overlap=0.5, batch_size=4, amp="bf16"):
    """img (1,3,H,W) in [0,1] on device -> list of maps [(p_wall,p_fib) latent, (p_wall,p_fib) per rater...]
    each (H,W) float32 tensors, Gaussian-blended."""
    _, _, H, W = img.shape
    ph, pw = max(0, window - H), max(0, window - W)
    if ph or pw:
        img = F.pad(img, (0, pw, 0, ph), mode="reflect")
    Hp, Wp = img.shape[-2:]
    stride = max(1, int(window * (1 - overlap)))
    coords = [(y, x) for y in _starts(Hp, window, stride) for x in _starts(Wp, window, stride)]
    g = _gauss(window, img.device)
    acc, wsum = None, torch.zeros(Hp, Wp, device=img.device)
    head, mode = model.head_type, model.label_mode
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    for i in range(0, len(coords), batch_size):
        cs = coords[i:i + batch_size]
        x = torch.cat([img[:, :, y:y + window, x0:x0 + window] for y, x0 in cs], 0)
        with torch.autocast(device_type=img.device.type, dtype=dt, enabled=dt is not None and img.device.type == "cuda"):
            out = model(x)
        maps = [wall_fib_probs(out["z"], head)]
        rp = rater_wall_fib(out, head, mode)
        if rp:
            maps += rp
        stacked = torch.stack([torch.stack(m, 1) for m in maps], 1).float()   # (b, M, 2, w, w)
        if acc is None:
            acc = torch.zeros((stacked.shape[1], 2, Hp, Wp), device=img.device)
        for k, (y, x0) in enumerate(cs):
            acc[:, :, y:y + window, x0:x0 + window] += stacked[k] * g
            wsum[y:y + window, x0:x0 + window] += g
    acc = acc / wsum
    return acc[:, :, :H, :W]          # (M,2,H,W)


@torch.no_grad()
def predict_image(models, img, icfg, amp="bf16"):
    """Returns mean maps (M,2,H,W) and per-(model,tta) maps list for spread estimation."""
    variants = []
    flips = TTA_FLIPS if icfg.tta else [()]
    for m in models:
        for fl in flips:
            x = img.flip(fl) if fl else img
            maps = sliding_window(m, x, icfg.window, icfg.overlap, icfg.batch_size, amp)
            if fl:
                maps = maps.flip([d for d in fl])          # (M,2,H,W): image dims are 2,3 again
            variants.append(maps)
    return variants


def evaluate_image(variants, lab: np.ndarray, head: str, n_models: int, surface=True) -> tuple[dict, np.ndarray]:
    d = decode_labels(lab)
    roi = d["roi"]
    experts = {"e1": (d["w1"] & roi, d["f1"] & roi), "e2": (d["w2"] & roi, d["f2"] & roi)}
    V = torch.stack(variants, 0).cpu().numpy()             # (V, M, 2, H, W)
    mean = V.mean(0)
    pw, pf = mean[0, 0], mean[0, 1]
    wall, fib = decode(pw, pf, head, roi)
    row = image_metrics(wall, fib, experts, surface=surface)
    roi_f = roi.astype(np.float32)
    row["FI_soft"] = float((pf * roi_f).sum() / max((pw * roi_f).sum(), 1e-6))
    # spread across TTA variants / ensemble members (latent FI)
    fis = np.array([fi(*decode(v[0, 0], v[0, 1], head, roi)) for v in V])
    lg = _logit01(fis)
    row["FI_var_std"] = float(np.nanstd(fis))
    row["FI_var_std_logit"] = float(np.nanstd(lg))
    if n_models > 1:
        row["FI_ens_std"] = float(np.nanstd(fis.reshape(n_models, -1).mean(1)))
        row["FI_ens_std_logit"] = float(np.nanstd(lg.reshape(n_models, -1).mean(1)))
    # expert reference values
    for k, (ew, ef) in experts.items():
        row[f"FI_{k}"] = fi(ew, ef)
        row[f"wall_area_{k}"] = float(ew.sum())
        row[f"fib_area_{k}"] = float(ef.sum())
    row["FI_mean"] = np.nanmean([row["FI_e1"], row["FI_e2"]])
    row["dice_wall_e1_e2"] = float(2 * (experts["e1"][0] & experts["e2"][0]).sum() /
                                   max(experts["e1"][0].sum() + experts["e2"][0].sum(), 1))
    row["dice_fib_e1_e2"] = float(2 * (experts["e1"][1] & experts["e2"][1]).sum() /
                                  max(experts["e1"][1].sum() + experts["e2"][1].sum(), 1)) \
        if (experts["e1"][1].sum() + experts["e2"][1].sum()) else 1.0
    preds_fib = [fib]
    if mean.shape[0] > 1:                                   # rater-specific heads
        preds_fib = []
        for r in range(1, mean.shape[0]):
            rw, rf = decode(mean[r, 0], mean[r, 1], head, roi)
            rr = image_metrics(rw, rf, experts, prefix=f"r{r}_", surface=False)
            row.update(rr)
            row[f"FI_soft_r{r}"] = float((mean[r, 1] * roi_f).sum() / max((mean[r, 0] * roi_f).sum(), 1e-6))
            preds_fib.append(rf)
        row["FI_rater_gap"] = abs(row["r1_FI_hat"] - row["r2_FI_hat"])
        row["FI_rater_gap_logit"] = float(abs(_logit01(row["r1_FI_hat"]) - _logit01(row["r2_FI_hat"])))
    row["ged_fib"] = ged(preds_fib, [experts["e1"][1], experts["e2"][1]])
    return row, mean


def save_probmap(path: Path, mean: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, maps=(np.clip(mean, 0, 1) * 255).astype(np.uint8))


def run_inference(models, meta: pd.DataFrame, cache_dir: Path, icfg, head: str, device, amp="bf16",
                  probmap_dir: Path | None = None, progress=True) -> pd.DataFrame:
    from tqdm import tqdm
    rows = []
    it = tqdm(meta.itertuples(), total=len(meta), desc="infer") if progress else meta.itertuples()
    for r in it:
        img = np.load(Path(cache_dir) / "img" / f"{r.stem}.npy")
        lab = np.load(Path(cache_dir) / "lab" / f"{r.stem}.npy")
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        variants = [v for v in predict_image(models, x, icfg, amp)]
        row, mean = evaluate_image(variants, lab, head, len(models))
        row.update({c: getattr(r, c) for c in ("file", "stem", "animal", "pc", "gn", "group", "role", "level")})
        rows.append(row)
        if probmap_dir is not None:
            save_probmap(Path(probmap_dir) / f"{r.stem}.npz", mean)
    return pd.DataFrame(rows)
