"""Evaluate a folder of predicted label maps (PNG, 0=bg, 1=wall, 2=fibrosis; any other value
treated as background) with exactly the same metrics as our models -> results/<name>/oof_predictions.csv.

python baselines/evaluate_masks.py --pred_dir D:/preds/nnunet_e1 --name nnunet_e1 [--mapping map.csv]
--mapping: optional csv with columns case_id,stem when files are not named <stem>.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frpv.config import load_config, resolve_path  # noqa: E402
from frpv.data.meta import load_meta  # noqa: E402
from frpv.eval.infer import evaluate_image  # noqa: E402


def evaluate_folder(pred_dir: Path, name: str, mapping: dict | None = None, head="nested"):
    cfg = load_config("configs/base.yaml")
    cache = resolve_path(cfg.paths.cache_dir)
    meta = load_meta(cache).set_index("stem")
    rows = []
    files = sorted(Path(pred_dir).glob("*.png"))
    for f in tqdm(files, desc=name):
        stem = mapping.get(f.stem, f.stem) if mapping else f.stem
        if stem not in meta.index:
            continue
        r = meta.loc[stem]
        lab_pred = np.array(Image.open(f))
        lab = np.load(cache / "lab" / f"{stem}.npy")
        assert lab_pred.shape == lab.shape, f"{f.name}: shape {lab_pred.shape} != {lab.shape}"
        pw = torch.from_numpy(((lab_pred == 1) | (lab_pred == 2)).astype(np.float32))
        pf = torch.from_numpy((lab_pred == 2).astype(np.float32))
        v = torch.stack([pw, pf], 0).unsqueeze(0)                     # (M=1,2,H,W)
        row, _ = evaluate_image([v], lab, head, 1)
        row.update(file=r.file, stem=stem, animal=r.animal, pc=r.pc, gn=r.gn, group=r.group, role=r.role,
                   level=r.level, fold=r.fold_official, exp=name)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = resolve_path(cfg.paths.results_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "oof_predictions.csv", index=False)
    print(f"{name}: {len(df)} images | dice_wall_e1={df.dice_wall_e1.mean():.3f} dice_fib_e1={df.dice_fib_e1.mean():.3f} "
          f"dice_fib_e2={df.dice_fib_e2.mean():.3f} FI_MAE(mean)={(df.FI_hat - df.FI_mean).abs().mean():.4f}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mapping", default=None)
    a = ap.parse_args()
    mp = None
    if a.mapping:
        m = pd.read_csv(a.mapping)
        mp = dict(zip(m.case_id, m.stem))
    evaluate_folder(Path(a.pred_dir), a.name, mp)
