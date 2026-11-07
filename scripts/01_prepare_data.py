"""Step 1 - index the dataset, build the memory-mapped cache, folds and 3-dye stain vectors.

python scripts/01_prepare_data.py                      # uses configs/base.yaml paths
python scripts/01_prepare_data.py --data_root D:/data/FRPV-PH

Outputs (cache/):
  img/*.npy, lab/*.npy, meta.csv, stain_vectors.json, data_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frpv.config import load_config, resolve_path  # noqa: E402
from frpv.data.meta import (BIT_F1, BIT_F2, BIT_ROI, BIT_W1, BIT_W2, assign_lopo,  # noqa: E402
                            find_data_root, make_official_folds, parse_filename)
from frpv.stain.physics import (DYES, estimate_stain_matrix, rgb_to_od,  # noqa: E402
                                save_stain_vectors)


def _mask(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L")) > 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--pixels_per_image", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = find_data_root(resolve_path(a.data_root or cfg.paths.data_root))
    cache = resolve_path(cfg.paths.cache_dir)
    (cache / "img").mkdir(parents=True, exist_ok=True)
    (cache / "lab").mkdir(parents=True, exist_ok=True)
    print(f"data root: {root}\ncache    : {cache}")

    files = sorted(p.name for p in (root / "raw_images").glob("*.png"))
    rng = np.random.default_rng(0)
    rows, od_samples, problems = [], {}, []
    for name in tqdm(files, desc="indexing"):
        info = parse_filename(name)
        stem = info["stem"]
        rgb = np.array(Image.open(root / "raw_images" / name).convert("RGB"))
        roi = _mask(root / "roi_masks" / name)
        w1, f1 = _mask(root / "expert_1" / "vascular_wall" / name), _mask(root / "expert_1" / "fibrosis" / name)
        w2, f2 = _mask(root / "expert_2" / "vascular_wall" / name), _mask(root / "expert_2" / "fibrosis" / name)
        shapes = {m.shape for m in (roi, w1, f1, w2, f2)} | {rgb.shape[:2]}
        if len(shapes) != 1:
            problems.append(f"{name}: shape mismatch {shapes}")
            continue
        if (f1 & ~w1).any() or (f2 & ~w2).any():
            problems.append(f"{name}: fibrosis outside wall")
        lab = (roi * BIT_ROI + (w1 & roi) * BIT_W1 + (f1 & roi) * BIT_F1 +
               (w2 & roi) * BIT_W2 + (f2 & roi) * BIT_F2).astype(np.uint8)
        ip, lp = cache / "img" / f"{stem}.npy", cache / "lab" / f"{stem}.npy"
        if a.force or not ip.exists():
            np.save(ip, rgb)
        if a.force or not lp.exists():
            np.save(lp, lab)
        w1r, f1r, w2r, f2r = w1 & roi, f1 & roi, w2 & roi, f2 & roi
        info.update(H=rgb.shape[0], W=rgb.shape[1], roi_area=int(roi.sum()),
                    wall_area_e1=int(w1r.sum()), fib_area_e1=int(f1r.sum()),
                    wall_area_e2=int(w2r.sum()), fib_area_e2=int(f2r.sum()))
        dsc = lambda x, y: 1.0 if (x.sum() + y.sum()) == 0 else float(2 * (x & y).sum() / (x.sum() + y.sum()))
        info["dice_wall_e1_e2"], info["dice_fib_e1_e2"] = dsc(w1r, w2r), dsc(f1r, f2r)
        info["FI_e1"] = info["fib_area_e1"] / max(info["wall_area_e1"], 1)
        info["FI_e2"] = info["fib_area_e2"] / max(info["wall_area_e2"], 1)
        rows.append(info)
        # pixels for stain estimation: tissue pixels inside ROI
        od = rgb_to_od(rgb[roi])
        od = od[od.sum(1) > 0.3]
        if len(od):
            sel = rng.choice(len(od), size=min(a.pixels_per_image, len(od)), replace=False)
            od_samples.setdefault(info["pc"], []).append(od[sel])

    meta = pd.DataFrame(rows)
    meta["FI_mean"] = (meta.FI_e1 + meta.FI_e2) / 2
    meta["fold_official"] = make_official_folds(meta, resolve_path(cfg.paths.official_splits))
    meta["fold_lopo"] = assign_lopo(meta)
    meta.to_csv(cache / "meta.csv", index=False)

    # ---- stain vectors (global + per project) and concentration statistics
    all_od = np.concatenate([np.concatenate(v) for v in od_samples.values()])
    sub = all_od[rng.choice(len(all_od), size=min(300_000, len(all_od)), replace=False)]
    m_global = estimate_stain_matrix(sub)
    per_proj = {pc: estimate_stain_matrix(np.concatenate(v)) for pc, v in od_samples.items()}
    conc = sub @ np.linalg.pinv(m_global)
    stats = {"mean": conc.mean(0).round(4).tolist(), "std": conc.std(0).round(4).tolist(),
             "per_project_mean": {pc: (np.concatenate(v) @ np.linalg.pinv(m_global)).mean(0).round(4).tolist()
                                  for pc, v in od_samples.items()}}
    save_stain_vectors(resolve_path(cfg.stain.vectors_file), m_global, per_proj, stats)

    rep = {"data_root": str(root), "n_images": len(meta), "n_animals": int(meta.animal.nunique()),
           "images_per_project": meta.pc.value_counts().to_dict(),
           "animals_per_project": meta.groupby("pc").animal.nunique().to_dict(),
           "image_sizes": meta.groupby(["H", "W"]).size().rename("n").reset_index().to_dict("records"),
           "problems": problems, "stain_matrix_global": {d: v for d, v in zip(DYES, m_global.round(4).tolist())},
           "concentration_stats": stats}
    (cache / "data_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("n_images", "n_animals", "images_per_project",
                                          "stain_matrix_global")}, indent=1))
    if problems:
        print(f"WARNING: {len(problems)} problems, see data_report.json")


if __name__ == "__main__":
    main()
