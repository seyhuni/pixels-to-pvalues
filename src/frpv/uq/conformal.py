"""Distribution-free intervals for the fibrosis index (FI) under clustered (animal-level) data.

Notation: images i nested in animals k; prediction f_i (model FI), reference y_i (expert FI),
heuristic scale s_i (model-derived difficulty), transform t (identity or logit).
Non-conformity score  S_i = |t(y_i) - t(f_i)| / s_i     (target 'both': max over the two experts).

Calibration schemes (all calibrate on whole animals; test animals are never seen):
  naive     : ordinary split-conformal on images (ignores clustering)
  pooled    : hierarchical conformal, pooled-CDF weights 1/(K n_k)  (Dunn, Wasserman & Ramdas;
              Lee, Barber & Willett) - targets coverage of a random image of a NEW animal
  subsample : one random image per calibration animal, repeated B times, majority vote
              (radius = median of per-draw quantiles)
  animal    : unit = animal; FI averaged over the animal's images (animal-level endpoint)
Mondrian (group-conditional) variants calibrate separately within each project.

Evaluation is by repeated random animal-level calibration/test partitions (stratified by project).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EPS_FI = 0.01            # FI is clipped to [0.01, 0.99] before the logit transform
P_FLOOR_DELTA = 0.02     # floor of p(1-p) argument in the delta-method fallback


def _logit(p):
    p = np.clip(p, EPS_FI, 1 - EPS_FI)
    return np.log(p / (1 - p))


def _expit(x):
    from scipy.special import expit
    return expit(x)


def _col(df, name, logit):
    """Scale column on the requested axis; '<name>_logit' is written by eval/infer.py."""
    if logit and f"{name}_logit" in df:
        return df[f"{name}_logit"].to_numpy(float), True
    return df[name].to_numpy(float), False


def _scale(df: pd.DataFrame, kind: str, transform: str = "identity") -> np.ndarray:
    """Heuristic difficulty scale s_i (label-free). For the logit transform, scales are taken
    directly on the logit axis when available (spread of logit-FI across TTA/ensemble members,
    logit gap between rater heads); otherwise a floored delta-method conversion is used.
    A soft floor s = sqrt(s^2 + c^2), c = 0.5 * median of positive raw scales, prevents
    zero-spread images from dominating the conformal quantile."""
    logit = transform == "logit"
    n = len(df)
    if kind == "const":
        return np.ones(n)
    parts, native = [], []
    comps = {"tta": ["FI_var_std"], "ens": ["FI_ens_std"], "rater": ["FI_rater_gap"],
             "combo": ["FI_var_std", "FI_rater_gap", "FI_ens_std"]}[kind]
    for c in comps:
        if c not in df or not df[c].notna().any():
            continue
        v, is_logit = _col(df, c, logit)
        if c == "FI_rater_gap":
            v = v / 2
        if logit and not is_logit:
            p = np.clip(df["FI_hat"].to_numpy(float) if "FI_hat" in df else 0.5, P_FLOOR_DELTA, 1 - P_FLOOR_DELTA)
            v = v / (p * (1 - p))
        parts.append(np.nan_to_num(v, nan=np.nan))
    if not parts:
        return np.ones(n)
    s = np.sqrt(np.nansum(np.square(np.stack(parts)), axis=0))
    pos = s[np.isfinite(s) & (s > 0)]
    c = 0.5 * float(np.median(pos)) if len(pos) else 1.0
    s = np.nan_to_num(s, nan=float(np.median(pos)) if len(pos) else 1.0)
    return np.sqrt(s ** 2 + c ** 2)


def available_scales(df):
    out = ["const"]
    if "FI_var_std" in df:
        out.append("tta")
    if "FI_ens_std" in df and df["FI_ens_std"].notna().any():
        out.append("ens")
    if "FI_rater_gap" in df and df["FI_rater_gap"].notna().any():
        out.append("rater")
    if "FI_var_std" in df:
        out.append("combo")
    return out


def prepare(df: pd.DataFrame, pred_col="FI_hat", scale="combo", transform="logit") -> pd.DataFrame:
    d = df[["animal", "pc", "FI_e1", "FI_e2", "FI_mean", pred_col]].copy()
    d = d.rename(columns={pred_col: "f"})
    d["s"] = _scale(df, scale, transform)                   # already on the transformed axis
    d = d[np.isfinite(d.f) & np.isfinite(d.FI_e1) & np.isfinite(d.FI_e2)].reset_index(drop=True)
    d["s_t"] = d.s
    tf = _logit if transform == "logit" else (lambda x: x)
    d["tf"] = tf(d.f)
    for c in ("FI_e1", "FI_e2", "FI_mean"):
        d["t_" + c] = tf(d[c])
    return d


def scores(d: pd.DataFrame, target: str) -> np.ndarray:
    if target == "both":
        r = np.maximum(np.abs(d.t_FI_e1 - d.tf), np.abs(d.t_FI_e2 - d.tf))
    else:
        r = np.abs(d["t_FI_" + target] - d.tf)
    return (r / d.s_t).to_numpy()


def covered(d: pd.DataFrame, target: str, q: float) -> np.ndarray:
    lo, hi = d.tf - q * d.s_t, d.tf + q * d.s_t
    if target == "both":
        return ((d.t_FI_e1 >= lo) & (d.t_FI_e1 <= hi) & (d.t_FI_e2 >= lo) & (d.t_FI_e2 <= hi)).to_numpy()
    y = d["t_FI_" + target]
    return ((y >= lo) & (y <= hi)).to_numpy()


def width_fi(d: pd.DataFrame, q: float, transform: str) -> np.ndarray:
    lo, hi = d.tf - q * d.s_t, d.tf + q * d.s_t
    if transform == "logit":
        return (_expit(hi) - _expit(lo)).to_numpy()
    return (np.clip(hi, 0, 1) - np.clip(lo, 0, 1)).to_numpy()


def conformal_quantile(s: np.ndarray, alpha: float, w: np.ndarray | None = None, n_units=None) -> float:
    s = np.asarray(s, float)
    if w is None:
        n = len(s)
        k = math.ceil((n + 1) * (1 - alpha))
        return float(np.sort(s)[k - 1]) if k <= n else math.inf
    K = n_units
    level = (1 - alpha) * (1 + 1 / K)
    if level >= 1:
        return math.inf
    o = np.argsort(s)
    cw = np.cumsum(w[o]) / w.sum()
    return float(s[o][np.searchsorted(cw, level)])


def calibrate(dc: pd.DataFrame, target: str, alpha: float, method: str, rng, B=50) -> float:
    s = scores(dc, target)
    if method == "naive":
        return conformal_quantile(s, alpha)
    if method == "pooled":
        n_k = dc.groupby("animal")["animal"].transform("count").to_numpy()
        return conformal_quantile(s, alpha, w=1.0 / n_k, n_units=dc.animal.nunique())
    if method == "subsample":
        groups = dc.groupby("animal").indices
        qs = []
        for _ in range(B):
            idx = [rng.choice(ix) for ix in groups.values()]
            qs.append(conformal_quantile(s[idx], alpha))
        return float(np.median(qs))
    raise ValueError(method)


def animal_level(d: pd.DataFrame, transform: str) -> pd.DataFrame:
    """Aggregate to animals: endpoint = mean FI over the animal's images; prediction = mean
    predicted FI; scale = RMS image scale / sqrt(n) (on the transformed axis)."""
    g = d.groupby("animal").agg(pc=("pc", "first"), f=("f", "mean"), FI_e1=("FI_e1", "mean"),
                                FI_e2=("FI_e2", "mean"), FI_mean=("FI_mean", "mean"),
                                s=("s", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                                n=("f", "size")).reset_index()
    g["s_t"] = g["s"] / np.sqrt(g["n"])
    tf = _logit if transform == "logit" else (lambda x: x)
    g["tf"] = tf(g.f)
    for c in ("FI_e1", "FI_e2", "FI_mean"):
        g["t_" + c] = tf(g[c])
    return g


def split_animals(animals_pc: pd.DataFrame, calib_frac, rng):
    calib = []
    for _, grp in animals_pc.groupby("pc"):
        a = grp.animal.to_numpy()
        n = max(1, int(round(calib_frac * len(a))))
        n = min(n, len(a) - 1) if len(a) > 1 else n
        calib += rng.choice(a, size=n, replace=False).tolist()
    return set(calib)


def evaluate(df: pd.DataFrame, target="mean", pred_col="FI_hat", scale="combo", transform="logit",
             method="pooled", alpha=0.1, mondrian=False, n_splits=500, calib_frac=0.5, seed=0,
             scheme="random") -> dict:
    """scheme='random': repeated random animal-level calibration/test partitions (stratified by project).
    scheme='loao'  : leave-one-animal-out - each animal is the test unit once and all other animals
                     (of the same project if mondrian) form the calibration set; every such split is a
                     valid split-conformal instance, so the average coverage is >= 1-alpha in expectation.
                     LOAO keeps per-project calibration sets as large as possible (Mondrian on small projects)."""
    rng = np.random.default_rng(seed)
    d = prepare(df, pred_col, scale, transform)
    if method == "animal":
        d = animal_level(d, transform)
    d = d.reset_index(drop=True)
    animals, a_idx = np.unique(d.animal.to_numpy(), return_inverse=True)
    a_pc = d.groupby("animal").pc.first().reindex(animals).to_numpy()
    pcs = np.array(sorted(set(a_pc)))
    nA = len(animals)
    cal_method = "naive" if method == "animal" else method
    covs, covs_img, widths, inf_frac, covs_fin = [], [], [], [], []
    per_pc = {pc: [] for pc in pcs}
    apc = pd.DataFrame({"animal": animals, "pc": a_pc})
    if scheme == "loao":
        splits = [set(animals) - {a} for a in animals]
    else:
        splits = (split_animals(apc, calib_frac, rng) for _ in range(n_splits))
    for cal in splits:
        is_cal_a = np.isin(animals, list(cal))
        is_cal = is_cal_a[a_idx]
        test = ~is_cal
        qs = np.full(len(d), np.nan)
        if mondrian:
            for pc in pcs:
                mc = is_cal & (a_pc[a_idx] == pc)
                mt = test & (a_pc[a_idx] == pc)
                if not mt.any():
                    continue
                q = calibrate(d[mc], target, alpha, cal_method, rng) if mc.any() else math.inf
                qs[mt] = q
        else:
            qs[test] = calibrate(d[is_cal], target, alpha, cal_method, rng)
        dt = d[test]
        qt = qs[test]
        fin = np.isfinite(qt)
        qf = np.where(fin, qt, 0.0)
        cov = np.where(fin, covered_q(dt, target, qf), True)
        wid = np.where(fin, width_q(dt, qf, transform), 1.0)
        ai = a_idx[test]
        cnt = np.bincount(ai, minlength=nA)
        has = cnt > 0
        a_cov = np.bincount(ai, weights=cov, minlength=nA)[has] / cnt[has]
        a_wid = np.bincount(ai, weights=wid, minlength=nA)[has] / cnt[has]
        a_covfin = np.bincount(ai, weights=(cov & fin).astype(float), minlength=nA)[has]
        nfin = np.bincount(ai, weights=fin.astype(float), minlength=nA)[has]
        if nfin.sum() > 0:
            covs_fin.append(float(np.sum(np.where(nfin > 0, a_covfin / np.maximum(nfin, 1), 0)) /
                                  max((nfin > 0).sum(), 1)))
        covs.append(a_cov.mean())
        covs_img.append(cov.mean())
        widths.append(a_wid.mean())
        inf_frac.append(1 - fin.mean())
        for pc in pcs:
            m = a_pc[has] == pc
            if m.any():
                per_pc[pc].append(a_cov[m].mean())
    res = dict(target=target, pred=pred_col, scale=scale, transform=transform, method=method,
               mondrian=mondrian, alpha=alpha, scheme=scheme, nominal=1 - alpha,
               coverage=np.mean(covs), coverage_sd=np.std(covs) if scheme != "loao" else np.nan,
               coverage_finite_only=np.mean(covs_fin) if covs_fin else np.nan,
               coverage_img_weighted=np.mean(covs_img), width=np.mean(widths),
               width_sd=np.std(widths), frac_infinite=np.mean(inf_frac),
               worst_project_cov=min(np.mean(v) for v in per_pc.values() if v))
    for pc, v in per_pc.items():
        res[f"cov_{pc}"] = np.mean(v) if v else np.nan
    return res


def covered_q(d, target, q):
    """Vectorised coverage with a per-row radius q."""
    lo, hi = d.tf.to_numpy() - q * d.s_t.to_numpy(), d.tf.to_numpy() + q * d.s_t.to_numpy()
    if target == "both":
        y1, y2 = d.t_FI_e1.to_numpy(), d.t_FI_e2.to_numpy()
        return (y1 >= lo) & (y1 <= hi) & (y2 >= lo) & (y2 <= hi)
    y = d["t_FI_" + target].to_numpy()
    return (y >= lo) & (y <= hi)


def width_q(d, q, transform):
    lo, hi = d.tf.to_numpy() - q * d.s_t.to_numpy(), d.tf.to_numpy() + q * d.s_t.to_numpy()
    if transform == "logit":
        return _expit(hi) - _expit(lo)
    return np.clip(hi, 0, 1) - np.clip(lo, 0, 1)


def interval_table(df, pred_col="FI_hat", scale="combo", transform="logit", alpha=0.1,
                   target="mean", method="pooled") -> pd.DataFrame:
    """Full-data calibrated intervals (all animals as calibration) for plotting; use only for
    illustration - coverage claims come from `evaluate`."""
    d = prepare(df, pred_col, scale, transform)
    q = calibrate(d, target, alpha, method, np.random.default_rng(0))
    lo, hi = d.tf - q * d.s_t, d.tf + q * d.s_t
    if transform == "logit":
        lo, hi = _expit(lo), _expit(hi)
    return d.assign(lo=np.clip(lo, 0, 1), hi=np.clip(hi, 0, 1), q=q)
