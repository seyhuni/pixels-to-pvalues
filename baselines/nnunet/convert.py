"""Convert the cache to nnU-Net v2 raw format (one dataset per expert) with the OFFICIAL folds.

  Dataset501_FRPVE1 : expert-1 labels      Dataset502_FRPVE2 : expert-2 labels
  labels: 0 background, 1 wall (without fibrosis), 2 fibrosis, 3 = ignore (outside ROI)

python baselines/nnunet/convert.py                 # write raw datasets + splits_final.json
python baselines/nnunet/convert.py --splits_only   # (re)write splits after plan_and_preprocess
Environment variables nnUNet_raw / nnUNet_preprocessed must be set (run_nnunet.bat does it).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from frpv.config import load_config, resolve_path  # noqa: E402
from frpv.data.meta import decode_labels, load_meta, to_categorical  # noqa: E402

DATASETS = {1: (501, "Dataset501_FRPVE1"), 2: (502, "Dataset502_FRPVE2")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_only", action="store_true")
    a = ap.parse_args()
    raw = Path(os.environ["nnUNet_raw"])
    pre = Path(os.environ["nnUNet_preprocessed"])
    cfg = load_config("configs/base.yaml")
    cache = resolve_path(cfg.paths.cache_dir)
    meta = load_meta(cache).reset_index(drop=True)
    meta["case_id"] = [f"FRPV_{i:04d}" for i in range(len(meta))]
    for e, (did, dname) in DATASETS.items():
        d = raw / dname
        if not a.splits_only:
            (d / "imagesTr").mkdir(parents=True, exist_ok=True)
            (d / "labelsTr").mkdir(parents=True, exist_ok=True)
            for r in tqdm(meta.itertuples(), total=len(meta), desc=dname):
                img = np.load(cache / "img" / f"{r.stem}.npy")
                lab = decode_labels(np.load(cache / "lab" / f"{r.stem}.npy"))
                y = to_categorical(lab[f"w{e}"], lab[f"f{e}"]).astype(np.uint8)
                y[~lab["roi"]] = 3
                ip = d / "imagesTr" / f"{r.case_id}_0000.png"
                if not ip.exists():
                    Image.fromarray(img).save(ip)
                Image.fromarray(y).save(d / "labelsTr" / f"{r.case_id}.png")
            ds = {"channel_names": {"0": "R", "1": "G", "2": "B"},
                  "labels": {"background": 0, "wall": 1, "fibrosis": 2, "ignore": 3},
                  "numTraining": len(meta), "file_ending": ".png",
                  "name": dname, "description": f"FRPV-PH expert {e} labels, ROI outside = ignore"}
            (d / "dataset.json").write_text(json.dumps(ds, indent=2))
            meta[["case_id", "stem", "file", "animal", "fold_official"]].to_csv(d / "case_mapping.csv", index=False)
        splits = []
        for k in sorted(meta.fold_official.unique()):
            splits.append({"train": meta.case_id[meta.fold_official != k].tolist(),
                           "val": meta.case_id[meta.fold_official == k].tolist()})
        (pre / dname).mkdir(parents=True, exist_ok=True)
        (pre / dname / "splits_final.json").write_text(json.dumps(splits, indent=1))
        print(f"{dname}: splits_final.json written ({len(splits)} official animal-level folds)")


if __name__ == "__main__":
    main()
