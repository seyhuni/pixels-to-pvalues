"""Step 6 - prediction-powered inference for treatment / disease effects on FI.

python scripts/06_ppi.py --exp final_ens
python scripts/06_ppi.py --exp final_ens --B 300 --nlab 1 2 3 5

Outputs results/<exp>/ppi/
  contrasts.csv : per pre-registered contrast x n_lab x estimator: coverage of the population effect,
                  CI width, bias, rejection rate (population bootstrap of animals)
  null.csv      : type-I error when both arms come from the same animal pool
  budget.csv    : CI width relative to 'classical' as a function of labelled images per animal
"""
import argparse

import pandas as pd
from tqdm import tqdm

import _common
from frpv.stats.analysis import CONTRASTS
from frpv.stats.ppi import null_calibration, simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--pred", default="FI_hat")
    ap.add_argument("--target", default="FI_mean", help="FI_mean | FI_e1 | FI_e2")
    ap.add_argument("--nlab", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    df = _common.load_oof(a.exp)
    df = df[df[a.pred].notna() & df[a.target].notna()]
    out = _common.results_dir() / a.exp / "ppi"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pc, ga, gb, desc in tqdm(CONTRASTS, desc="contrasts"):
        A = df[(df.pc == pc) & (df.gn == ga)].animal.unique()
        Bn = df[(df.pc == pc) & (df.gn == gb)].animal.unique()
        if len(A) < 2 or len(Bn) < 2:
            continue
        for n in a.nlab:
            r = simulate(df, A, Bn, a.target, a.pred, n_lab=n, B=a.B, alpha=a.alpha, seed=n)
            r["contrast"], r["pc"] = desc, pc
            rows.append(r)
    c = pd.concat(rows, ignore_index=True)
    c.to_csv(out / "contrasts.csv", index=False)
    nulls = []
    pool = df.animal.unique()
    for n_g in (5, 10):
        for n in a.nlab:
            nulls.append(null_calibration(df, pool, n_g, a.target, a.pred, n_lab=n, B=a.B * 2, alpha=a.alpha, seed=n))
    nu = pd.concat(nulls, ignore_index=True)
    nu.to_csv(out / "null.csv", index=False)
    w = c.pivot_table(index=["contrast", "n_lab"], columns="estimator", values="width").reset_index()
    for e in ["ppi", "ppi_tuned", "naive", "oracle"]:
        if e in w:
            w[f"{e}_rel_width"] = w[e] / w["classical"]
    w.to_csv(out / "budget.csv", index=False)
    print(c.groupby(["estimator", "n_lab"])[["coverage", "width"]].mean().round(3).to_string())
    print(nu.groupby(["estimator", "n_lab"]).type1.mean().round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
