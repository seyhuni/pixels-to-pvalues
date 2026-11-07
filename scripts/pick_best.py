"""Select the best stage-1 backbone WITHOUT looking at test folds: criterion = inner-validation
metrics (last validation row of each fold's train_log.csv, averaged over folds):
  rank(mean inner-val fibrosis Dice over both experts) + rank(inner-val FI MAE).

python scripts/pick_best.py --prefix s1_ --verbose      -> prints e.g. fm:uni2h
python scripts/pick_best.py --prefix s1_ --field run     -> s1_uni2h
python scripts/pick_best.py --encoder fm:uni2h --field run   (run name of a given encoder)
python scripts/pick_best.py --encoder fm:uni2h --field tag   (short tag used in stage-2..4 run names)
"""
import argparse

import pandas as pd

import _common
from frpv.config import load_config


def enc_tag(spec: str) -> str:
    name = spec.split(":", 1)[-1]
    return name.split(".")[0].replace("/", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="s1_")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--field", default="encoder", choices=["encoder", "run", "tag"])
    a = ap.parse_args()
    if a.encoder and a.field == "tag":
        print(enc_tag(a.encoder))
        return
    res = _common.results_dir()
    rows = []
    for d in sorted(res.glob(f"{a.prefix}*")):
        if not (d / "config.yaml").exists():
            continue
        cfg = load_config(d / "config.yaml")
        if a.encoder and cfg.model.encoder != a.encoder:
            continue
        logs = sorted(d.glob("fold*/train_log.csv"))
        folds_done = len(list(d.glob("fold*/predictions.csv")))
        vals = []
        for lp in logs:
            t = pd.read_csv(lp)
            if "val_dice_fib_e1" in t:
                last = t.dropna(subset=["val_dice_fib_e1"]).iloc[-1]
                vals.append(dict(dice=(last.val_dice_fib_e1 + last.val_dice_fib_e2) / 2, mae=last.val_FI_mae_mean))
        if not vals:
            continue
        v = pd.DataFrame(vals).mean()
        rows.append(dict(run=d.name, encoder=cfg.model.encoder, tag=enc_tag(cfg.model.encoder),
                         folds_done=folds_done, val_dice_fib=v.dice, val_fi_mae=v.mae))
    if not rows:
        raise SystemExit("no stage-1 runs with inner-validation logs found")
    t = pd.DataFrame(rows)
    complete = t[t.folds_done >= 5]
    if len(complete) and not a.encoder:
        t = complete
    t["rank"] = t.val_dice_fib.rank(ascending=False) + t.val_fi_mae.rank(ascending=True)
    t = t.sort_values(["rank", "val_fi_mae"])
    if a.verbose:
        print(t.to_string(index=False))
    print(t[a.field].iloc[0])


if __name__ == "__main__":
    main()
