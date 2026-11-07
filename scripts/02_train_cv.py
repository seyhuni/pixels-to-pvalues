"""Step 2 - train + out-of-fold inference for one experiment (all folds of the chosen split).

python scripts/02_train_cv.py --exp configs/exp/main_uni2h.yaml
python scripts/02_train_cv.py --exp configs/exp/main_uni2h.yaml --folds 0 1 train.epochs=2 exp_name=smoke
Re-running skips folds whose predictions.csv already exists (resume-safe); use --force to retrain.
"""
import argparse

import _common  # noqa: F401
from frpv.config import load_config
from frpv.train import collect_oof, run_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("overrides", nargs="*", help="key=value overrides, e.g. train.epochs=10")
    a = ap.parse_args()
    cfg = load_config(a.base, a.exp, overrides=a.overrides)
    folds = a.folds if a.folds else list(cfg.data.folds)
    if cfg.data.split == "lopo" and folds == [0, 1, 2, 3, 4]:
        folds = [0, 1, 2, 3]
    print(f"== experiment {cfg.exp_name} | encoder {cfg.model.encoder} | head {cfg.model.head} | "
          f"labels {cfg.model.label_mode} | stain aug {cfg.stain.augment} / branch {cfg.stain.branch_input} "
          f"| split {cfg.data.split} folds {folds}")
    for k in folds:
        run_fold(cfg, k, skip_if_done=not a.force)
    df = collect_oof(cfg)
    if df is not None:
        print(f"OOF ({len(df)} images): dice_wall_e1={df.dice_wall_e1.mean():.3f} "
              f"dice_fib_e1={df.dice_fib_e1.mean():.3f} dice_fib_e2={df.dice_fib_e2.mean():.3f} "
              f"FI_MAE(mean)={(df.FI_hat - df.FI_mean).abs().mean():.4f}")


if __name__ == "__main__":
    main()
