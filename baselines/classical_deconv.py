"""Classical baseline: colour deconvolution + threshold on the aniline-blue (collagen) channel inside
the EXPERT vessel wall (i.e. an oracle wall - this mirrors the manual Fiji workflow).

Two variants (written as separate result folders):
  classical_otsu    : per-image Otsu threshold (training-free)
  classical_cvthr   : one global threshold tuned on the training folds (min FI MAE vs expert mean)

python baselines/classical_deconv.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frpv.config import load_config, resolve_path  # noqa: E402
from frpv.data.meta import decode_labels, load_meta  # noqa: E402
from frpv.eval.metrics import dice, fi  # noqa: E402
from frpv.stain.physics import deconvolve, load_stain_matrix  # noqa: E402


def main():
    cfg = load_config("configs/base.yaml")
    cache = resolve_path(cfg.paths.cache_dir)
    meta = load_meta(cache)
    m = load_stain_matrix(resolve_path(cfg.stain.vectors_file))
    thr_grid = np.linspace(0.05, 1.2, 47)
    recs, curves = [], []
    for r in tqdm(meta.itertuples(), total=len(meta), desc="deconv"):
        img = np.load(cache / "img" / f"{r.stem}.npy")
        d = decode_labels(np.load(cache / "lab" / f"{r.stem}.npy"))
        blue = deconvolve(img, m)[..., 0]
        row = dict(file=r.file, stem=r.stem, animal=r.animal, pc=r.pc, gn=r.gn, group=r.group, role=r.role,
                   level=r.level, fold=r.fold_official)
        for k in ("1", "2"):
            w, f = d["w" + k] & d["roi"], d["f" + k] & d["roi"]
            row[f"FI_e{k}"] = fi(w, f)
            vals = blue[w]
            t = threshold_otsu(vals) if len(vals) > 10 else np.inf
            pf = w & (blue > t)
            row[f"otsu_FI_w{k}"] = fi(w, pf)
            row[f"otsu_dice_fib_e{k}"] = dice(pf, f)
            # FI and Dice as a function of a global threshold (for CV tuning), vectorised in the wall
            fv = f[w]
            pred = vals[None, :] > thr_grid[:, None]
            ps = pred.sum(1)
            fis = ps / max(len(vals), 1)
            dices = np.where(ps + fv.sum() > 0, 2 * (pred & fv[None]).sum(1) / np.maximum(ps + fv.sum(), 1), 1.0)
            curves.append(dict(stem=r.stem, k=k, fis=fis.tolist(), dices=dices.tolist()))
        recs.append(row)
    df = pd.DataFrame(recs)
    df["FI_mean"] = (df.FI_e1 + df.FI_e2) / 2
    res = resolve_path(cfg.paths.results_dir)
    # ---- variant 1: Otsu
    o = df.copy()
    o["FI_hat"] = (o.otsu_FI_w1 + o.otsu_FI_w2) / 2
    o["dice_fib_e1"], o["dice_fib_e2"] = o.otsu_dice_fib_e1, o.otsu_dice_fib_e2
    o["dice_wall_e1"] = o["dice_wall_e2"] = 1.0
    o["containment_viol"] = 0.0
    o["dice_fib_e1_e2"] = np.nan
    _save(res / "classical_otsu", o)
    # ---- variant 2: CV-tuned global threshold
    cv = {(c["stem"], c["k"]): c for c in curves}
    out = df.copy()
    for k_fold in sorted(df.fold.unique()):
        tr = df[df.fold != k_fold]
        err = np.zeros(len(thr_grid))
        for r in tr.itertuples():
            fis = (np.array(cv[(r.stem, "1")]["fis"]) + np.array(cv[(r.stem, "2")]["fis"])) / 2
            err += np.abs(fis - r.FI_mean)
        j = int(np.argmin(err))
        te = df.fold == k_fold
        for i in np.where(te)[0]:
            s = df.stem.iloc[i]
            out.loc[out.index[i], "FI_hat"] = (cv[(s, "1")]["fis"][j] + cv[(s, "2")]["fis"][j]) / 2
            out.loc[out.index[i], "dice_fib_e1"] = cv[(s, "1")]["dices"][j]
            out.loc[out.index[i], "dice_fib_e2"] = cv[(s, "2")]["dices"][j]
            out.loc[out.index[i], "thr"] = thr_grid[j]
    out["dice_wall_e1"] = out["dice_wall_e2"] = 1.0
    out["containment_viol"] = 0.0
    out["dice_fib_e1_e2"] = np.nan
    _save(res / "classical_cvthr", out)


def _save(d: Path, df: pd.DataFrame):
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "oof_predictions.csv", index=False)
    print(f"{d.name}: dice_fib_e1={df.dice_fib_e1.mean():.3f} dice_fib_e2={df.dice_fib_e2.mean():.3f} "
          f"FI_MAE(mean)={(df.FI_hat - df.FI_mean).abs().mean():.4f}  (ORACLE expert wall)")


if __name__ == "__main__":
    main()
