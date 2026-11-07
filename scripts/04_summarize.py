"""Step 4 - summary tables over all experiments (segmentation + fibrosis-index agreement).

python scripts/04_summarize.py                       # every results/*/oof_predictions.csv
python scripts/04_summarize.py --ref a2_dual --B 2000  # + paired animal-bootstrap tests vs a reference

Writes results/summary/{summary.csv, summary.md, paired_vs_<ref>.csv, human_reference.csv}
Metrics are per-image means (ROI-restricted); CIs = animal-level percentile bootstrap.
"""
import argparse
import warnings

import numpy as np
import pandas as pd

import _common
from frpv.stats.analysis import agreement, animal_bootstrap, animal_means, paired_animal_bootstrap


def row_for(name, df, B):
    r = {"exp": name, "n_images": len(df), "n_animals": df.animal.nunique()}
    df = df.copy()
    df["dice_fib_avg"] = (df.dice_fib_e1 + df.dice_fib_e2) / 2
    df["dice_wall_avg"] = (df.dice_wall_e1 + df.dice_wall_e2) / 2
    df["abs_err_mean"] = (df.FI_hat - df.FI_mean).abs()
    for c in ["dice_wall_e1", "dice_wall_e2", "dice_fib_e1", "dice_fib_e2", "msd_wall_e1", "msd_fib_e1", "ged_fib"]:
        if c in df:
            r[c] = df[c].mean()
    r["containment_viol_img_frac"] = float((df.containment_viol > 0).mean())
    for key, col in [("dice_fib_avg", "dice_fib_avg"), ("FI_MAE_mean", "abs_err_mean")]:
        est, lo, hi = animal_bootstrap(df, lambda d, c=col: d[c].mean(), B=B)
        r[key], r[key + "_lo"], r[key + "_hi"] = est, lo, hi
    for ref in ["e1", "e2", "mean"]:
        ag = agreement(df.FI_hat, df[f"FI_{ref}"])
        r[f"FI_MAE_{ref}"], r[f"FI_bias_{ref}"] = ag["mae"], ag["bias"]
        r[f"FI_CCC_{ref}"], r[f"FI_ICC_{ref}"] = ag["ccc"], ag["icc"]
    am = animal_means(df, ["FI_hat", "FI_mean", "FI_e1", "FI_e2"])
    ag = agreement(am.FI_hat, am.FI_mean)
    r["animal_FI_MAE_mean"], r["animal_FI_CCC_mean"], r["animal_FI_r_mean"] = ag["mae"], ag["ccc"], ag["pearson"]
    if "r1_FI_hat" in df:
        # style identifiability: does head r reproduce expert r better than the other expert?
        r["r1_dice_fib_e1"], r["r1_dice_fib_e2"] = df.r1_dice_fib_e1.mean(), df.r1_dice_fib_e2.mean()
        r["r2_dice_fib_e1"], r["r2_dice_fib_e2"] = df.r2_dice_fib_e1.mean(), df.r2_dice_fib_e2.mean()
        r["r1_FI_MAE_e1"] = (df.r1_FI_hat - df.FI_e1).abs().mean()
        r["r2_FI_MAE_e2"] = (df.r2_FI_hat - df.FI_e2).abs().mean()
    return r


def human_reference(df):
    """Expert-2 treated as a 'model' evaluated against expert 1 (the inter-rater ceiling)."""
    r = {"exp": "HUMAN (E2 vs E1)", "n_images": len(df), "n_animals": df.animal.nunique()}
    if "dice_wall_e1_e2" in df:
        r["dice_wall_e1"] = df.dice_wall_e1_e2.mean()
        r["msd_note"] = "see dataset paper Table 3"
    r["dice_fib_e1"] = df.dice_fib_e1_e2.mean()
    ag = agreement(df.FI_e2, df.FI_e1)
    r.update(FI_MAE_e1=ag["mae"], FI_bias_e1=ag["bias"], FI_CCC_e1=ag["ccc"], FI_ICC_e1=ag["icc"])
    am = animal_means(df, ["FI_e1", "FI_e2"])
    ag = agreement(am.FI_e2, am.FI_e1)
    r.update(animal_FI_MAE_e1=ag["mae"], animal_FI_CCC_e1=ag["ccc"], animal_FI_r_e1=ag["pearson"])
    return r


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return adj


def constant_reference(df):
    """Predict, for each test fold, the mean expert FI of the other folds (no image information)."""
    pred = df.FI_mean.copy()
    for k in df.fold.unique():
        pred[df.fold == k] = df.FI_mean[df.fold != k].mean()
    ag = agreement(pred, df.FI_mean)
    return {"exp": "CONSTANT (train-fold mean FI)", "n_images": len(df), "FI_MAE_mean": ag["mae"],
            "FI_CCC_mean": ag["ccc"]}


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    res = _common.results_dir()
    names = a.exps or sorted(p.parent.name for p in res.glob("*/oof_predictions.csv"))
    out = res / "summary"
    out.mkdir(parents=True, exist_ok=True)
    rows, dfs = [], {}
    for n in names:
        df = pd.read_csv(res / n / "oof_predictions.csv")
        dfs[n] = df
        rows.append(row_for(n, df, a.B))
        print(f"{n:28s} n={len(df):4d} diceFib(avg)={rows[-1]['dice_fib_avg']:.3f} "
              f"FI_MAE(mean)={rows[-1]['FI_MAE_mean']:.4f} CCC={rows[-1]['FI_CCC_mean']:.3f}")
    from frpv.config import load_config, resolve_path
    from frpv.data.meta import load_meta
    meta = load_meta(resolve_path(load_config("configs/base.yaml").paths.cache_dir))
    meta["fold"] = meta.fold_official
    if "dice_fib_e1_e2" in meta:
        hr = pd.DataFrame([human_reference(meta), constant_reference(meta)])
        hr.to_csv(out / "human_reference.csv", index=False)
        print(hr.T.to_string())
    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)
    cols = [c for c in ["exp", "dice_wall_e1", "dice_fib_e1", "dice_fib_e2", "dice_fib_avg", "ged_fib",
                        "FI_MAE_mean", "FI_CCC_mean", "animal_FI_CCC_mean", "containment_viol_img_frac"] if c in s]
    try:
        md = s[cols].round(4).to_markdown(index=False)
    except ImportError:
        md = s[cols].round(4).to_string(index=False)
    (out / "summary.md").write_text(md if len(s) else "")
    if a.ref and a.ref in dfs:
        pr = []
        for n, df in dfs.items():
            if n == a.ref:
                continue
            for col in ["dice_fib_e1", "dice_fib_e2"]:
                d, lo, hi, p = paired_animal_bootstrap(dfs[a.ref], df, col, B=a.B)
                pr.append(dict(exp=n, ref=a.ref, metric=col, diff=d, lo=lo, hi=hi, p=p))
            x = df.assign(abs_err=(df.FI_hat - df.FI_mean).abs())
            y = dfs[a.ref].assign(abs_err=(dfs[a.ref].FI_hat - dfs[a.ref].FI_mean).abs())
            d, lo, hi, p = paired_animal_bootstrap(y, x, "abs_err", B=a.B)
            pr.append(dict(exp=n, ref=a.ref, metric="FI_abs_err_mean", diff=d, lo=lo, hi=hi, p=p))
        prd = pd.DataFrame(pr)
        if len(prd):   # Holm correction within each metric family
            prd["p_holm"] = prd.groupby("metric").p.transform(holm)
        prd.to_csv(out / f"paired_vs_{a.ref}.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
