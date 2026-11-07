"""Agreement statistics, animal-level bootstrap, mixed models and rater-dependence analyses."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Pre-registered contrasts (project, reference group, comparison group, description)
CONTRASTS = [
    ("M", 62, 61, "CTEPH model vs healthy (M)"),
    ("RE", 0, 1, "CTEPH model vs healthy (RE)"),
    ("AH", 0, 1, "PAH model vs healthy (AH)"),
    ("RE", 1, 2, "Treatment 2 vs untreated (RE)"),
    ("RE", 1, 3, "Treatment 3 vs untreated (RE)"),
    ("AH", 1, 2, "Treatment 2 vs untreated (AH)"),
    ("AH", 1, 3, "Treatment 3 vs untreated (AH)"),
    ("RA", 3, 1, "Treatment 1 vs untreated (RA)"),
    ("RA", 3, 2, "Treatment 2 vs untreated (RA)"),
    ("RA", 3, 4, "Treatment 4 vs untreated (RA)"),
]


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    sxy = np.cov(x, y, ddof=0)[0, 1]
    return float(2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def icc_a1(x, y):
    """ICC(A,1) two-way, absolute agreement, single measures (McGraw & Wong)."""
    X = np.c_[x, y].astype(float)
    X = X[np.isfinite(X).all(1)]
    n, k = X.shape
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def agreement(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    d = x - y
    return dict(mae=float(np.abs(d).mean()), bias=float(d.mean()), loa=float(1.96 * d.std(ddof=1)),
                pearson=float(np.corrcoef(x, y)[0, 1]), ccc=ccc(x, y), icc=icc_a1(x, y), n=int(m.sum()))


def animal_means(df: pd.DataFrame, cols) -> pd.DataFrame:
    keep = ["animal", "pc", "gn", "group", "role"]
    keep = [c for c in keep if c in df]
    agg = {c: "mean" for c in cols}
    agg.update({c: "first" for c in keep if c != "animal"})
    return df.groupby("animal").agg(agg).reset_index()


def animal_bootstrap(df: pd.DataFrame, fn, B=1000, seed=0, alpha=0.05):
    """Percentile CI of statistic fn(df) resampling animals with replacement."""
    rng = np.random.default_rng(seed)
    groups = {a: t for a, t in df.groupby("animal")}
    keys = list(groups)
    vals = []
    for _ in range(B):
        pick = rng.choice(keys, size=len(keys), replace=True)
        vals.append(fn(pd.concat([groups[k] for k in pick], ignore_index=True)))
    vals = np.array(vals)
    return float(fn(df)), float(np.nanquantile(vals, alpha / 2)), float(np.nanquantile(vals, 1 - alpha / 2))


def paired_animal_bootstrap(a: pd.DataFrame, b: pd.DataFrame, col, B=2000, seed=0):
    """Difference in mean per-image metric `col` between two experiments, resampling animals.
    Returns diff, lo, hi, two-sided bootstrap p."""
    m = a[["file", "animal", col]].merge(b[["file", col]], on="file", suffixes=("_a", "_b"))
    m["d"] = m[col + "_b"] - m[col + "_a"]
    per = m.groupby("animal").d.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    obs = per["sum"].sum() / per["count"].sum()
    boots = []
    idx = np.arange(len(per))
    for _ in range(B):
        s = per.iloc[rng.choice(idx, len(idx))]
        boots.append(s["sum"].sum() / s["count"].sum())
    boots = np.array(boots)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(obs), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), float(p)


def mixed_effect(df: pd.DataFrame, value_col: str, pc: str, ga: int, gb: int) -> dict:
    """Random-intercept (animal) linear mixed model value ~ arm; fallback Welch on animal means."""
    d = df[(df.pc == pc) & (df.gn.isin([ga, gb]))].copy()
    d = d[np.isfinite(d[value_col])]
    d["arm"] = (d.gn == gb).astype(int)
    out = dict(pc=pc, ref=ga, cmp=gb, measure=value_col, n_animals=d.animal.nunique(), n_images=len(d))
    try:
        import statsmodels.formula.api as smf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm(f"{value_col} ~ arm", d, groups=d["animal"]).fit(reml=True)
        ci = m.conf_int().loc["arm"].to_numpy()
        out.update(effect=float(m.params["arm"]), lo=float(ci[0]), hi=float(ci[1]),
                   p=float(m.pvalues["arm"]), method="LMM")
    except Exception:
        from .ppi import welch_ci
        am = d.groupby("animal").agg(v=(value_col, "mean"), arm=("arm", "first"))
        e, lo, hi, p = welch_ci(am.v[am.arm == 0], am.v[am.arm == 1])
        out.update(effect=e, lo=lo, hi=hi, p=p, method="Welch(animal means)")
    return out


def rater_dependence(df: pd.DataFrame, measures, alpha=0.05) -> pd.DataFrame:
    rows = []
    for pc, ga, gb, desc in CONTRASTS:
        if not ((df.pc == pc) & (df.gn == ga)).any() or not ((df.pc == pc) & (df.gn == gb)).any():
            continue
        for meas in measures:
            if meas not in df:
                continue
            r = mixed_effect(df, meas, pc, ga, gb)
            r["contrast"] = desc
            r["significant"] = r["p"] < alpha
            rows.append(r)
    t = pd.DataFrame(rows)
    if len(t):
        flip = t.groupby("contrast").significant.agg(lambda s: s.nunique() > 1).rename("conclusion_depends_on_measure")
        t = t.merge(flip, on="contrast")
    return t


def rater_bias_by_project(df: pd.DataFrame) -> pd.DataFrame:
    """Observed expert bias (E2-E1) vs model-predicted rater bias (r2-r1) per project."""
    rows = []
    for pc, t in df.groupby("pc"):
        r = dict(pc=pc, n=len(t), obs_FI_bias=float((t.FI_e2 - t.FI_e1).mean()),
                 obs_fib_area_ratio=float(t.fib_area_e2.sum() / max(t.fib_area_e1.sum(), 1)),
                 obs_wall_area_ratio=float(t.wall_area_e2.sum() / max(t.wall_area_e1.sum(), 1)))
        if "r2_FI_hat" in t:
            r.update(pred_FI_bias=float((t.r2_FI_hat - t.r1_FI_hat).mean()),
                     pred_fib_area_ratio=float(t.r2_fib_area.sum() / max(t.r1_fib_area.sum(), 1)),
                     pred_wall_area_ratio=float(t.r2_wall_area.sum() / max(t.r1_wall_area.sum(), 1)))
        rows.append(r)
    out = pd.DataFrame(rows)
    if "pred_FI_bias" in out and len(out) > 2:
        out.attrs["corr_bias"] = float(np.corrcoef(out.obs_FI_bias, out.pred_FI_bias)[0, 1])
    return out


def image_level_bias_corr(df: pd.DataFrame) -> dict:
    """Does the model predict *which images* the experts disagree on (and in which direction)?"""
    if "r2_FI_hat" not in df:
        return {}
    obs = (df.FI_e2 - df.FI_e1).to_numpy()
    pred = (df.r2_FI_hat - df.r1_FI_hat).to_numpy()
    m = np.isfinite(obs) & np.isfinite(pred)
    from scipy.stats import spearmanr
    rho = spearmanr(obs[m], pred[m]).correlation
    gap_obs = np.abs(obs[m])
    gap_pred = df.FI_rater_gap.to_numpy()[m]
    rho_gap = spearmanr(gap_obs, gap_pred).correlation
    return dict(spearman_signed_bias=float(rho), spearman_abs_gap=float(rho_gap), n=int(m.sum()))
