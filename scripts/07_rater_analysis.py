"""Step 7 - "does the conclusion depend on the rater?" and "does the model learn rater style?"

python scripts/07_rater_analysis.py --exp final_ens
python scripts/07_rater_analysis.py --experts_only      # motivating analysis, no model needed

Outputs results/<exp>/rater/ (or results/experts_only/rater/)
  dependence.csv     : mixed-model effect / CI / p for every contrast measured with
                       E1, E2, mean(E1,E2), model latent, model rater-1 head, model rater-2 head
  bias_by_project.csv: observed E2-E1 bias vs model-predicted r2-r1 bias per project
  image_bias.json    : image-level Spearman correlation between observed and predicted rater gaps
"""
import argparse
import json

import pandas as pd

import _common
from frpv.config import load_config, resolve_path
from frpv.data.meta import load_meta
from frpv.stats.analysis import image_level_bias_corr, rater_bias_by_project, rater_dependence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--experts_only", action="store_true")
    a = ap.parse_args()
    if a.experts_only or a.exp is None:
        cfg = load_config("configs/base.yaml")
        df = load_meta(resolve_path(cfg.paths.cache_dir))
        name = "experts_only"
    else:
        df = _common.load_oof(a.exp)
        name = a.exp
    out = _common.results_dir() / name / "rater"
    out.mkdir(parents=True, exist_ok=True)
    measures = ["FI_e1", "FI_e2", "FI_mean", "FI_hat", "r1_FI_hat", "r2_FI_hat"]
    dep = rater_dependence(df, measures)
    dep.to_csv(out / "dependence.csv", index=False)
    piv = dep.pivot_table(index="contrast", columns="measure", values="p")
    print("p-values per contrast and measurement source:\n", piv.round(3).to_string())
    eff = dep.pivot_table(index="contrast", columns="measure", values="effect")
    print("\neffects:\n", eff.round(4).to_string())
    bp = rater_bias_by_project(df)
    bp.to_csv(out / "bias_by_project.csv", index=False)
    print("\n", bp.round(3).to_string(index=False))
    ib = image_level_bias_corr(df)
    ib["project_level_corr"] = bp.attrs.get("corr_bias")
    (out / "image_bias.json").write_text(json.dumps(ib, indent=2))
    print(ib)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
