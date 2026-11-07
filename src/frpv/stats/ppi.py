"""Prediction-powered inference (PPI / PPI++) for clustered preclinical designs.

Setting: animals k in groups g; each animal has n_k vessel-segment images; the model predicts FI
for ALL images (f_i), experts label only a random subset L_k (|L_k| = n_lab per animal).
Animal-level target: theta_k = mean_i y_i over all images of animal k.

Per-animal PPI estimator (unbiased for theta_k for any fixed lambda, because L_k is a simple
random subsample of the animal's images):
    theta_hat_k(lambda) = lambda * mean_{i in k} f_i + mean_{i in L_k} (y_i - lambda f_i)
Group means and the group difference are computed on theta_hat_k; the CI uses the
between-animal variability (Welch t), which automatically contains the within-animal
labelling noise -> valid with animals as the independent units.
lambda = 1 is classical PPI (Angelopoulos et al., Science 2023); 'tuned' is a PPI++-style
power-tuned lambda (cf. PPI++, Angelopoulos et al., arXiv:2311.01453) adapted to clustered
subsampling: the variance-optimal lambda for a per-animal estimator is the WITHIN-animal slope of y on
f; it is fitted leave-one-animal-out so the per-animal estimates stay independent; clipped to [0,1].

Estimators compared: classical (labelled only), naive (predictions only), ppi, ppi_tuned, oracle (all labels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ESTIMATORS = ["classical", "naive", "ppi", "ppi_tuned", "oracle"]


def welch_ci(a: np.ndarray, b: np.ndarray, alpha=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = (a.var(ddof=1) if na > 1 else np.nan), (b.var(ddof=1) if nb > 1 else np.nan)
    diff = b.mean() - a.mean()
    se2 = va / na + vb / nb
    se = np.sqrt(se2)
    dfree = se2 ** 2 / ((va / na) ** 2 / max(na - 1, 1) + (vb / nb) ** 2 / max(nb - 1, 1)) if se2 > 0 else 1
    tq = stats.t.ppf(1 - alpha / 2, dfree)
    p = 2 * (1 - stats.t.cdf(abs(diff) / se, dfree)) if se > 0 else np.nan
    return diff, diff - tq * se, diff + tq * se, p


def one_sample_ci(a, alpha=0.05):
    a = np.asarray(a, float)
    m, se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
    tq = stats.t.ppf(1 - alpha / 2, len(a) - 1)
    return m, m - tq * se, m + tq * se


def within_slope(y, f, unit):
    """Within-unit regression slope of y on f (unit-centred) - the variance-optimal lambda for
    per-animal subsampling; falls back to the pooled slope when units have < 2 labels."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    df = pd.DataFrame({"y": y, "f": f, "u": unit})
    g = df.groupby("u")
    yc, fc = y - g.y.transform("mean").to_numpy(), f - g.f.transform("mean").to_numpy()
    den = float((fc ** 2).sum())
    if den <= 1e-12:
        if len(y) < 3 or np.var(f) <= 0:
            return 1.0
        return float(np.clip(np.cov(y, f)[0, 1] / np.var(f, ddof=1), 0.0, 1.0))
    return float(np.clip((yc * fc).sum() / den, 0.0, 1.0))


def animal_estimates(images: pd.DataFrame, lab_mask: np.ndarray, target: str, pred: str, lam) -> pd.Series:
    """Per-unit PPI estimates. lam: float, or 'loao' = within-unit slope fitted on the labelled
    images of all OTHER units (keeps units independent)."""
    y, f = images[target].to_numpy(float), images[pred].to_numpy(float)
    unit = images["unit"].to_numpy()
    out = {}
    for u in np.unique(unit):
        m = unit == u
        lm = m & lab_mask
        if isinstance(lam, str):
            other = lab_mask & ~m
            lu = within_slope(y[other], f[other], unit[other])
        else:
            lu = lam
        out[u] = lu * f[m].mean() + (y[lm] - lu * f[lm]).mean()
    return pd.Series(out)


def _choose(pool, design, frac, rng):
    pool = list(pool)
    if design == "bootstrap":
        return list(rng.choice(pool, size=len(pool), replace=True))
    if design == "subsample":
        m = max(2, int(round(frac * len(pool))))
        m = min(m, len(pool))
        return list(rng.choice(pool, size=m, replace=False))
    return pool                                            # 'fixed'


def _draw_study(df, animals_a, animals_b, n_lab, rng, design="subsample", frac=0.7):
    """Return image table for one simulated study and the labelled mask.
    design: 'subsample' (default; animals drawn WITHOUT replacement, no duplicate units),
            'bootstrap' (with replacement), 'fixed' (all animals, only label randomness).
    `df` may be a DataFrame or a dict {animal: DataFrame} (faster)."""
    by = df if isinstance(df, dict) else {a: t for a, t in df.groupby("animal")}
    parts, uid = [], 0
    for grp_name, pool in (("A", animals_a), ("B", animals_b)):
        chosen = _choose(pool, design, frac, rng)
        for a in chosen:
            t = by[a].copy()
            t["unit"], t["arm"] = f"{grp_name}{uid}", grp_name
            uid += 1
            parts.append(t)
    st = pd.concat(parts, ignore_index=True)
    lab = np.zeros(len(st), bool)
    for u, ix in st.groupby("unit").indices.items():
        k = len(ix) if n_lab is None else min(n_lab, len(ix))
        lab[rng.choice(ix, size=k, replace=False)] = True
    return st, lab


def estimate_difference(st, lab, target, pred, alpha=0.05):
    arm = st.groupby("unit")["arm"].first()
    res = {}
    all_lab = np.ones(len(st), bool)
    for name in ESTIMATORS:
        if name == "classical":
            th = animal_estimates(st, lab, target, pred, lam=0.0)
        elif name == "naive":
            th = st.groupby("unit")[pred].mean()
        elif name == "ppi":
            th = animal_estimates(st, lab, target, pred, lam=1.0)
        elif name == "ppi_tuned":
            th = animal_estimates(st, lab, target, pred, lam="loao")
        else:
            th = animal_estimates(st, all_lab, target, pred, lam=0.0)
        a, b = th[arm[th.index] == "A"].to_numpy(), th[arm[th.index] == "B"].to_numpy()
        res[name] = welch_ci(a, b, alpha)
    return res


def population_difference(df, animals_a, animals_b, target):
    th = df.groupby("animal")[target].mean()
    return th[list(animals_b)].mean() - th[list(animals_a)].mean()


def simulate(df: pd.DataFrame, animals_a, animals_b, target="FI_mean", pred="FI_hat", n_lab=2,
             B=500, alpha=0.05, design="subsample", frac=0.7, seed=0) -> pd.DataFrame:
    """Coverage / width / rejection rate of each estimator over B simulated studies.
    The observed animals of each arm are the (finite) population; the estimand is the population
    difference of animal-level mean FI computed with ALL expert labels. Each simulated study draws
    animals (see _draw_study) and labels n_lab random segments per animal."""
    rng = np.random.default_rng(seed)
    truth = population_difference(df, animals_a, animals_b, target)
    by = {a: t for a, t in df.groupby("animal")}
    recs = []
    for b in range(B):
        st, lab = _draw_study(by, list(animals_a), list(animals_b), n_lab, rng, design, frac)
        for name, (d, lo, hi, p) in estimate_difference(st, lab, target, pred, alpha).items():
            recs.append(dict(rep=b, estimator=name, diff=d, lo=lo, hi=hi, p=p,
                             covered=(lo <= truth <= hi), width=hi - lo, reject=p < alpha if np.isfinite(p) else False))
    r = pd.DataFrame(recs)
    out = r.groupby("estimator").agg(coverage=("covered", "mean"), width=("width", "mean"),
                                     bias=("diff", lambda x: float(np.mean(x) - truth)),
                                     reject_rate=("reject", "mean")).reset_index()
    out["truth"] = truth
    out["n_lab"] = n_lab if n_lab is not None else -1
    return out


def null_calibration(df: pd.DataFrame, pool, n_per_group, target="FI_mean", pred="FI_hat", n_lab=2,
                     B=1000, alpha=0.05, seed=0) -> pd.DataFrame:
    """Both arms are disjoint random subsets (without replacement) of the SAME animal pool -> the
    population difference is 0; reject_rate estimates the type-I error of each estimator."""
    rng = np.random.default_rng(seed)
    pool = list(pool)
    by = {a: t for a, t in df.groupby("animal")}
    recs = []
    for b in range(B):
        ab = rng.choice(pool, size=2 * n_per_group, replace=False)
        st, lab = _draw_study(by, list(ab[:n_per_group]), list(ab[n_per_group:]), n_lab, rng, design="fixed")
        for name, (d, lo, hi, p) in estimate_difference(st, lab, target, pred, alpha).items():
            recs.append(dict(estimator=name, covered=lo <= 0 <= hi, width=hi - lo,
                             reject=p < alpha if np.isfinite(p) else False))
    r = pd.DataFrame(recs)
    out = r.groupby("estimator").agg(type1=("reject", "mean"), coverage0=("covered", "mean"),
                                     width=("width", "mean")).reset_index()
    out["n_lab"], out["n_per_group"] = n_lab, n_per_group
    return out
