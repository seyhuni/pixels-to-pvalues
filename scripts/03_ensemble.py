"""Step 3 (optional) - deep-ensemble inference over several trained runs (e.g. seeds).

python scripts/03_ensemble.py --runs s1_uni2h e4_seed1 e4_seed2 --name final_ens
Writes results/<name>/fold*/predictions.csv (+ FI_ens_std column) and oof_predictions.csv.
All runs must share the split and the architecture-defining options of the first run is
re-read from each run's own config.yaml (so mixing heads is allowed but not recommended).
"""
import argparse

import pandas as pd

import _common
from frpv.config import load_config, resolve_path, save_config
from frpv.data.meta import fold_column, load_meta
from frpv.eval.infer import run_inference
from frpv.train import get_device, load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="model_last.pt")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--max_images", type=int, default=None, help="debug only")
    a = ap.parse_args()
    res = _common.results_dir()
    cfgs = [load_config(res / r / "config.yaml") for r in a.runs]
    c0 = cfgs[0]
    assert len({c.data.split for c in cfgs}) == 1, "runs use different splits"
    heads = {c.model.head for c in cfgs}
    assert len(heads) == 1, "ensemble members must share the head type"
    assert len({c.model.label_mode for c in cfgs}) == 1, "ensemble members must share label_mode"
    device = get_device()
    cache = resolve_path(c0.paths.cache_dir)
    meta = load_meta(cache)
    fcol = fold_column(c0.data.split)
    out = res / a.name
    out.mkdir(parents=True, exist_ok=True)
    save_config(dict(c0, exp_name=a.name, ensemble_of=a.runs), out / "config.yaml")
    icfg = type(c0.infer)(dict(c0.infer, tta=not a.no_tta))
    parts = []
    for k in sorted(meta[fcol].unique()):
        ck = [res / r / f"fold{k}" / a.ckpt for r in a.runs]
        if not all(p.exists() for p in ck):
            print(f"fold {k}: missing checkpoints, skipped")
            continue
        p_csv = out / f"fold{k}" / "predictions.csv"
        if p_csv.exists():
            parts.append(pd.read_csv(p_csv))
            continue
        models = [load_checkpoint(c, p, device) for c, p in zip(cfgs, ck)]
        amp = c0.train.amp if device.type == "cuda" else "none"
        mk = meta[meta[fcol] == k]
        if a.max_images:
            mk = mk.head(a.max_images)
        df = run_inference(models, mk, cache, icfg, c0.model.head, device, amp)
        df.insert(0, "fold", k)
        df.insert(1, "exp", a.name)
        p_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p_csv, index=False)
        parts.append(df)
        del models
    if parts:
        oof = pd.concat(parts, ignore_index=True)
        oof.to_csv(out / "oof_predictions.csv", index=False)
        print(f"ensemble OOF: dice_fib_e1={oof.dice_fib_e1.mean():.3f} FI_MAE={(oof.FI_hat-oof.FI_mean).abs().mean():.4f}")


if __name__ == "__main__":
    main()
