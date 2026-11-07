"""Collect nnU-Net out-of-fold validation predictions (fold_k/validation/*.png) and evaluate them.

python baselines/nnunet/collect.py      -> results/nnunet_e1, results/nnunet_e2
"""
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from evaluate_masks import evaluate_folder  # noqa: E402
from convert import DATASETS  # noqa: E402

TRAINER = os.environ.get("NNUNET_TRAINER_DIR", "nnUNetTrainer__nnUNetPlans__2d")


def main():
    raw, res = Path(os.environ["nnUNet_raw"]), Path(os.environ["nnUNet_results"])
    for e, (_, dname) in DATASETS.items():
        mapping = pd.read_csv(raw / dname / "case_mapping.csv")
        mp = dict(zip(mapping.case_id, mapping.stem))
        tmp = res / dname / "_oof_all"
        tmp.mkdir(parents=True, exist_ok=True)
        n = 0
        for k in range(5):
            vd = res / dname / TRAINER / f"fold_{k}" / "validation"
            for f in vd.glob("*.png"):
                shutil.copy(f, tmp / f.name)
                n += 1
        print(f"{dname}: {n} OOF predictions")
        if n:
            evaluate_folder(tmp, f"nnunet_e{e}", mp, head="flat")


if __name__ == "__main__":
    main()
