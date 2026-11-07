"""Step 5 - conformal intervals for the fibrosis index (FI) with animal-level guarantees.

python scripts/05_conformal.py --exp final_ens
python scripts/05_conformal.py --exp s1_uni2h --splits 300 --alphas 0.1 0.2

Grid: target {e1,e2,mean,both} x scale {const,tta,ens,rater,combo} x transform {identity,logit}
      x method {naive,pooled,subsample,animal} x mondrian {no,yes} x alpha.
Writes results/<exp>/conformal/{grid.csv, intervals_full.csv}
Key questions answered by grid.csv:
  * Is image-exchangeable ('naive') calibration miscalibrated at the animal level vs 'pooled'?
  * Do rater-aware scales (predicted inter-rater gap) give narrower intervals at equal coverage?
  * Does the logit transform (ratio-aware, bounded) shorten intervals for low-FI images?
  * Does Mondrian (per-project) calibration fix the worst-project coverage under rater/stain shift?
    (always read together with frac_infinite: small projects can yield infinite = trivial intervals;
     the 'loao' scheme maximises per-project calibration size)
"""
import argparse
import itertools

import pandas as pd
from tqdm import tqdm

import _common
from frpv.uq.conformal import available_scales, evaluate, interval_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--pred", default="FI_hat", help="FI_hat (hard masks) or FI_soft")
    ap.add_argument("--splits", type=int, default=500)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--targets", nargs="+", default=["mean", "e1", "e2", "both"])
    ap.add_argument("--methods", nargs="+", default=["naive", "pooled", "subsample", "animal"])
    ap.add_argument("--schemes", nargs="+", default=["random", "loao"])
    ap.add_argument("--quick", action="store_true", help="small grid for testing")
    a = ap.parse_args()
    df = _common.load_oof(a.exp)
    out = _common.results_dir() / a.exp / "conformal"
    out.mkdir(parents=True, exist_ok=True)
    scales = available_scales(df)
    transforms = ["identity", "logit"]
    if a.quick:
        scales, transforms, a.targets, a.methods = scales[:2], ["logit"], ["mean"], ["naive", "pooled", "animal"]
    grid = list(itertools.product(a.targets, scales, transforms, a.methods, [False, True], a.alphas, a.schemes))
    rows = []
    for tgt, sc, tr, me, mo, al, sch in tqdm(grid, desc="conformal grid"):
        rows.append(evaluate(df, tgt, a.pred, sc, tr, me, al, mo, n_splits=a.splits, scheme=sch))
    g = pd.DataFrame(rows)
    g.to_csv(out / "grid.csv", index=False)
    # illustration: full-data calibrated intervals with the default configuration
    it = interval_table(df, a.pred, scale=scales[-1], transform="logit", alpha=0.1, target="mean", method="pooled")
    it.to_csv(out / "intervals_full.csv", index=False)
    show = g[(g.alpha == a.alphas[0]) & (g.target == "mean")]
    print(show[["scheme", "mondrian", "scale", "transform", "method", "coverage", "coverage_finite_only",
                "frac_infinite", "worst_project_cov", "width"]]
          .round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
