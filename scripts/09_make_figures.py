"""Paper figures (Figures 1-6). Usage:
    python scripts/09_make_figures.py results cache/meta.csv data/FRPV-PH figures_out
Output file names follow the analysis order; mapping to the paper: fig1->Figure 1, fig3_rater_dependence->Figure 2,
fig2_method->Figure 3, fig6_lopo->Figure 4, fig4_conformal->Figure 5, fig5_ppi->Figure 6."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image
from scipy import ndimage

R, META, DATA, OUT = map(Path, sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)

# ---- style (validated categorical slots, light surface) -------------------
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "savefig.dpi": 400, "figure.dpi": 100, "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
})
W = 6.1  # inches (15.5 cm text width of the IOP single-column template)


def grid_y(ax):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)


def grid_x(ax):
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)


def panel(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


meta = pd.read_csv(META)
meta["d_FI"] = meta.FI_e2 - meta.FI_e1

# =========================================================================
# Figure 1 - data and inter-rater disagreement
# =========================================================================
def load_mask(sub, f):
    return np.array(Image.open(DATA / sub / f).convert("L")) > 127


def contour(m):
    return m & ~ndimage.binary_erosion(m, iterations=1)


def overlay(f):
    img = np.array(Image.open(DATA / "raw_images" / f).convert("RGB")).astype(float) / 255
    roi = load_mask("roi_masks", f)
    w1, w2 = load_mask("expert_1/vascular_wall", f) & roi, load_mask("expert_2/vascular_wall", f) & roi
    f1, f2 = load_mask("expert_1/fibrosis", f) & roi, load_mask("expert_2/fibrosis", f) & roi
    out = img.copy()
    out[~roi] = out[~roi] * 0.45 + 0.55            # fade outside ROI
    def paint(m, hexc):
        c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255
        out[m] = c
    paint(contour(w1), "#2b2b2b")
    paint(contour(f1), BLUE)
    paint(contour(f2), ORANGE)
    # zoomed crop (aspect 1.6) centred on the densest region of expert disagreement inside the wall
    dis = ndimage.uniform_filter(((f1 ^ f2) & (w1 | w2)).astype(float), 201)
    cy, cx = np.unravel_index(np.argmax(dis), dis.shape)
    ch, cw = 360, 576
    y0 = int(np.clip(cy - ch // 2, 0, roi.shape[0] - ch)); x0 = int(np.clip(cx - cw // 2, 0, roi.shape[1] - cw))
    return out[y0:y0 + ch, x0:x0 + cw]


m_cand = meta[(meta.group == "M61")].assign(k=lambda d: (d.d_FI - d.d_FI.median()).abs()).sort_values("k")
re_cand = meta[(meta.group == "RE1")].assign(k=lambda d: (d.dice_fib_e1_e2 - 0.68).abs() + d.d_FI.abs() + (d.FI_mean - 0.16).abs()).sort_values("k")
ex = [m_cand.iloc[0], re_cand.iloc[0]]
json.dump({"fig1_examples": [r.file for r in ex]}, open(OUT / "fig1_examples.json", "w"))

fig = plt.figure(figsize=(W, 4.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.12)
for i, r in enumerate(ex):
    ax = fig.add_subplot(gs[0, i]); ax.imshow(overlay(r.file)); ax.set_axis_off()
    ax.set_title(f"$\\bf{{({'ab'[i]})}}$  {r.group} ({r.disease} {r.role}): FI$_{{E1}}$ = {r.FI_e1:.3f}, FI$_{{E2}}$ = {r.FI_e2:.3f}",
                 fontsize=7, loc="left", color=INK)
h = [Line2D([], [], color="#3a3a3a", lw=1.5, label="Vascular wall (E1)"),
     Line2D([], [], color=BLUE, lw=1.5, label="Fibrosis, expert E1"),
     Line2D([], [], color=ORANGE, lw=1.5, label="Fibrosis, expert E2")]
fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.555), fontsize=6.8)

# (c) per-project distribution of E2 - E1 FI difference
ax = fig.add_subplot(gs[1, 0]); panel(ax, "c")
order = ["AH", "M", "RA", "RE"]
rng = np.random.default_rng(0)
for k, pc in enumerate(order):
    d = meta.loc[meta.pc == pc, "d_FI"].to_numpy()
    ax.scatter(k + rng.uniform(-0.22, 0.22, len(d)), d, s=3, color=MUTED, alpha=0.55, lw=0)
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    ax.plot([k - 0.3, k + 0.3], [med, med], color=INK, lw=1.6)
    ax.plot([k, k], [q1, q3], color=INK, lw=0.9)
    ax.text(k, 0.33, f"n={len(d)}", ha="center", fontsize=6.3, color=INK2)
ax.axhline(0, color=AXIS, lw=0.8)
ax.set_xticks(range(4), order); ax.set_ylabel("FI$_{E2}$ − FI$_{E1}$ (image level)")
ax.set_ylim(-0.3, 0.36); grid_y(ax)
ax.set_xlabel("Project")

# (d) animal-level agreement of FI between the two experts
ax = fig.add_subplot(gs[1, 1]); panel(ax, "d", x=-0.3)
an = meta.groupby(["animal", "pc"], as_index=False)[["FI_e1", "FI_e2"]].mean()
mk = {"AH": "o", "M": "s", "RA": "^", "RE": "D"}
for pc in order:
    s = an[an.pc == pc]
    ax.scatter(s.FI_e1, s.FI_e2, s=13, marker=mk[pc], facecolor="white" if pc != "M" else ORANGE,
               edgecolor=INK2 if pc != "M" else ORANGE, lw=0.7, label=pc)
lim = [0, 0.3]
ax.plot(lim, lim, color=AXIS, lw=0.8, ls="--")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("Animal-level FI, expert E1"); ax.set_ylabel("Animal-level FI, expert E2")
ax.legend(title="Project", loc="upper left", handletextpad=0.2, title_fontsize=6.8)
grid_y(ax)
fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.08)
save(fig, "fig1_data_disagreement")

# =========================================================================
# Figure 2 - method schematic
# =========================================================================
fig, ax = plt.subplots(figsize=(W, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.set_axis_off()


def box(x, y, w, h, title, sub=None, fc="#f4f4f2", ec=AXIS):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", fc=fc, ec=ec, lw=0.8))
    if sub:
        ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="center", fontsize=6.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 4.6, sub, ha="center", va="top", fontsize=5.6, color=INK2, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=6.5, fontweight="bold", color=INK)


def arrow(p, q, c=INK2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=c, shrinkA=0, shrinkB=0))


LB, LO, LA, LV = "#e3eefb", "#fdeae1", "#dff4ec", "#e9e6f9"
box(0.5, 19, 12, 12, "RGB image", "0.252 µm/px\nROI only")
box(16, 33, 22, 12, "Pathology FM encoder", "UNI2-h (ViT-H/14) + LoRA\ninput resampled to 0.5 µm/px", fc=LB, ec=BLUE)
box(16, 4, 22, 14, "Beer–Lambert layer", "OD = −log$_{10}$(I/I$_0$) = C S$^{\\top}$\nS: aniline blue, acid fuchsin,\norange G (learnable residual)", fc=LO, ec=ORANGE)
box(41.5, 33, 18, 12, "DPT-style decoder", "4 ViT blocks\nreassembled, multi-scale")
box(41.5, 4, 18, 14, "Detail branch", "dye concentrations C\nstrides 1, 2, 4, 8\n(native resolution)", fc=LO, ec=ORANGE)
box(63, 19, 11.5, 12, "Fusion", "native\nresolution")
box(78, 34, 21.5, 12, "Nested head", "p(wall),  p(fib | wall)\nfib $\\subseteq$ wall by construction", fc=LA, ec=AQUA)
box(78, 18.5, 21.5, 12, "Rater offsets", "logit$_r$ = logit + δ$_r$,  Σ$_r$ δ$_r$ = 0\nL2 + TV regularised", fc=LA, ec=AQUA)
box(78, 2.5, 21.5, 12.5, "Inference on FI", "animal-level conformal\nintervals; PPI++ contrasts", fc=LV, ec=VIOLET)
arrow((12.9, 27), (15.6, 38)); arrow((12.9, 23), (15.6, 12))
arrow((38.4, 39), (41.1, 39)); arrow((38.4, 11), (41.1, 11))
arrow((59.9, 38), (64.5, 31.4)); arrow((59.9, 12), (64.5, 18.6))
arrow((74.9, 27.5), (77.6, 38)); arrow((74.9, 23.5), (77.6, 24.5))
arrow((88.75, 33.6), (88.75, 30.9)); arrow((88.75, 18.1), (88.75, 15.4))
ax.text(16, 48, "Hierarchy-consistent, rater-aware segmentation network", fontsize=6.6, color=INK2, style="italic")
ax.text(16, 0.6, "Stain physics", fontsize=6.0, color=ORANGE)
save(fig, "fig2_method")

# =========================================================================
# Figure 3 - scientific conclusions depend on the rater
# =========================================================================
dep = pd.read_csv(R / "final_ens_uni2h/rater/dependence.csv")
contr = ["CTEPH model vs healthy (M)", "Treatment 3 vs untreated (AH)", "Treatment 3 vs untreated (RE)",
         "Treatment 2 vs untreated (AH)", "PAH model vs healthy (AH)", "Treatment 4 vs untreated (RA)"]
meas = [("FI_e1", "Expert E1", BLUE, "o", True), ("FI_e2", "Expert E2", ORANGE, "o", True),
        ("FI_mean", "Expert mean", INK2, "o", True), ("r1_FI_hat", "Model, E1 head", BLUE, "o", False),
        ("r2_FI_hat", "Model, E2 head", ORANGE, "o", False), ("FI_hat", "Model, consensus", INK2, "D", False)]
fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw=dict(width_ratios=[1, 1.0], wspace=0.08))
for j, (ax, cs) in enumerate(zip(axes, [contr[:3], contr[3:]])):
    ys = []
    for i, c in enumerate(cs):
        for k, (m, lab, col, mk_, filled) in enumerate(meas):
            r = dep[(dep.contrast == c) & (dep.measure == m)].iloc[0]
            y = i * 8 + k
            ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.0)
            ax.scatter(r.effect, y, s=16, marker=mk_, color=col if filled else "white", edgecolor=col, lw=1.0, zorder=3)
            if r.p < 0.05:
                ax.text(r.hi + 0.004, y, "*", va="center", fontsize=8, color=INK)
        ys.append(i * 8 + 2.5)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(ys, [c.replace(" vs ", "\nvs ") for c in cs]); ax.invert_yaxis()
    ax.set_xlabel("Group difference in FI (LMM, 95% CI)")
    grid_x(ax); ax.tick_params(axis="y", length=0)
    if j == 1:
        ax.yaxis.tick_right(); ax.spines["left"].set_visible(False)
    panel(ax, "ab"[j], x=-0.02 if j == 0 else 0.0, y=1.02)
h = [Line2D([], [], color=c, marker=mk_, mfc=c if f else "white", mec=c, lw=1.0, ms=4, label=l) for _, l, c, mk_, f in meas]
fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.11), fontsize=6.5, columnspacing=1.0, handlelength=1.6)
save(fig, "fig3_rater_dependence")

# =========================================================================
# Figure 4 - conformal intervals: coverage vs width
# =========================================================================
g = pd.read_csv(R / "final_ens_uni2h/conformal/grid.csv")
g = g[(g.target == "mean") & (g.alpha == 0.1) & (g["transform"] == "identity") & (g.scheme == "random") & (~g.mondrian)]
scales = [("const", "Constant", MUTED), ("tta", "TTA spread", AQUA), ("ens", "Ensemble spread", VIOLET),
          ("rater", "Rater-head gap", ORANGE), ("combo", "Combined", BLUE)]
meths = [("naive", "Naive (image)", "o"), ("pooled", "Hierarchical pooled (image)", "s"), ("animal", "Animal level", "D")]
fig, axes = plt.subplots(1, 2, figsize=(W, 2.55), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.32))
ax = axes[0]; panel(ax, "a", x=-0.16)
for s, sl, sc in scales:
    for m, ml, mk_ in meths:
        r = g[(g.scale == s) & (g.method == m)].iloc[0]
        ax.scatter(r.width, r.worst_project_cov, s=24, marker=mk_, color=sc, edgecolor="white", lw=0.6, zorder=3)
ax.axhline(0.9, color=INK2, lw=0.8, ls="--"); ax.text(0.176, 0.905, "nominal 0.90", fontsize=6.3, color=INK2, ha="right", va="bottom")
ax.set_xlabel("Mean interval width (FI units)"); ax.set_ylabel("Worst-project coverage")
ax.set_ylim(0.55, 0.95); grid_y(ax)
h1 = [Line2D([], [], ls="", marker="o", color=c, ms=4.5, label=l) for _, l, c in scales]
h2 = [Line2D([], [], ls="", marker=mk_, color=INK2, mfc="white", ms=4.5, label=l) for _, l, mk_ in meths]
ax.legend(handles=h1 + h2, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=5.9, ncol=3, handletextpad=0.1, columnspacing=0.8)
ax = axes[1]; panel(ax, "b", x=-0.2)
pcs = ["AH", "M", "RA", "RE"]
for k, (m, s, lab, col, off) in enumerate([("pooled", "const", "Image, constant", MUTED, -0.27),
                                           ("pooled", "rater", "Image, rater gap", ORANGE, -0.09),
                                           ("animal", "const", "Animal, constant", MUTED, 0.09),
                                           ("animal", "rater", "Animal, rater gap", ORANGE, 0.27)]):
    r = g[(g.scale == s) & (g.method == m)].iloc[0]
    v = [r[f"cov_{p}"] for p in pcs]
    mk_ = "s" if m == "pooled" else "D"
    ax.scatter(np.arange(4) + off, v, s=20, marker=mk_, color=col, edgecolor="white", lw=0.5, zorder=3, label=lab)
ax.axhline(0.9, color=INK2, lw=0.8, ls="--")
ax.set_xticks(range(4), pcs); ax.set_xlabel("Project"); ax.set_ylabel("Coverage")
ax.set_ylim(0.55, 1.02); grid_y(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=5.9, ncol=2, handletextpad=0.1, columnspacing=0.8)
save(fig, "fig4_conformal")

# =========================================================================
# Figure 5 - prediction-powered inference
# =========================================================================
ct = pd.read_csv(R / "final_ens_uni2h/ppi/contrasts.csv")
nl = pd.read_csv(R / "final_ens_uni2h/ppi/null.csv")
bud = pd.read_csv(R / "final_ens_uni2h/ppi/budget.csv")
est = [("classical", "Classical (labels only)", MUTED, "o"), ("naive", "Naive plug-in (model only)", ORANGE, "s"),
       ("ppi", "PPI", AQUA, "^"), ("ppi_tuned", "PPI++", BLUE, "D")]
fig, axes = plt.subplots(1, 3, figsize=(W, 2.35), gridspec_kw=dict(wspace=0.45))
ax = axes[0]; panel(ax, "a", x=-0.3)
c1 = ct[ct.n_lab == 1]
cl = list(dict.fromkeys(c1.contrast))
for k, (e, lab, col, mk_) in enumerate(est):
    v = [c1[(c1.contrast == c) & (c1.estimator == e)].bias.iloc[0] for c in cl]
    ax.scatter(v, np.arange(len(cl)) + (k - 1.5) * 0.17, s=14, marker=mk_, color=col, edgecolor="white", lw=0.4, zorder=3)
ax.axvline(0, color=AXIS, lw=0.8)
ax.set_yticks(range(len(cl)), [c.replace("Treatment", "T").replace(" vs untreated", " vs untr.").replace(" vs healthy", " vs healthy") for c in cl], fontsize=5.8)
ax.invert_yaxis(); ax.set_xlabel("Bias of estimated contrast"); grid_x(ax); ax.tick_params(axis="y", length=0)
ax = axes[1]; panel(ax, "b", x=-0.3)
for e, lab, col, mk_ in est[1:]:
    col_ = {"naive": "naive_rel_width", "ppi": "ppi_rel_width", "ppi_tuned": "ppi_tuned_rel_width"}[e]
    s = bud.groupby("n_lab")[col_].mean()
    ax.plot(s.index, s.values, color=col, marker=mk_, ms=4, mfc=col, mec="white", mew=0.4, ls="--" if e == "naive" else "-")
ax.axhline(1, color=MUTED, lw=0.9, ls="--")
ax.set_xticks([1, 2, 3, 5]); ax.set_xlabel("Labelled segments per animal"); ax.set_ylabel("CI width / classical")
grid_y(ax)
ax = axes[2]; panel(ax, "c", x=-0.3)
for e, lab, col, mk_ in est:
    s = nl[nl.estimator == e].groupby("n_lab").type1.mean()
    ax.plot(s.index, s.values, color=col, marker=mk_, ms=4, mfc=col, mec="white", mew=0.4, label=lab, ls="--" if e == "naive" else "-")
ax.axhline(0.05, color=INK2, lw=0.8, ls="--")
ax.set_ylim(0, 0.1); ax.set_xticks([1, 2, 3, 5]); ax.set_xlabel("Labelled segments per animal"); ax.set_ylabel("Type-I error (null)")
grid_y(ax)
h = [Line2D([], [], color=c, marker=m_, ls="--" if k == "naive" else "-", ms=4, label=l) for k, l, c, m_ in est]
fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.15), fontsize=6.5)
save(fig, "fig5_ppi")

# =========================================================================
# Figure 6 - leave-one-project-out
# =========================================================================
def per_proj(e):
    d = pd.read_csv(R / e / "oof_predictions.csv")
    d["pc"] = d.file.str.extract(r"^([A-Z]+)")[0]
    d["dfa"] = (d.dice_fib_e1 + d.dice_fib_e2) / 2
    d["ae"] = (d.FI_hat - d.FI_mean).abs()
    d["gap"] = d.r2_FI_hat - d.r1_FI_hat if "r2_FI_hat" in d else np.nan
    return d.groupby("pc").agg(dfa=("dfa", "mean"), ae=("ae", "mean"), gap=("gap", "mean"))


cv, lf, ld = per_proj("final_ens_uni2h"), per_proj("l3_full_uni2h"), per_proj("l3_dual_uni2h")
hum = meta.groupby("pc").dice_fib_e1_e2.mean()
obs_gap = meta.groupby("pc").d_FI.mean()
fig, axes = plt.subplots(1, 3, figsize=(W, 2.2), gridspec_kw=dict(wspace=0.42))
x = np.arange(4)
series = [("Official CV (ensemble)", cv, BLUE, "o", -0.2), ("LOPO, proposed", lf, ORANGE, "s", 0.0), ("LOPO, dual-loss", ld, MUTED, "^", 0.2)]
for j, (col_, yl) in enumerate([("dfa", "Fibrosis Dice (mean of E1, E2)"), ("ae", "FI absolute error vs expert mean")]):
    ax = axes[j]; panel(ax, "ab"[j], x=-0.3)
    for lab, d, c, mk_, off in series:
        ax.scatter(x + off, d.loc[pcs, col_], s=20, marker=mk_, color=c, edgecolor="white", lw=0.4, zorder=3, label=lab)
    if col_ == "dfa":
        ax.scatter(x, hum.loc[pcs], s=40, marker="_", color=INK, lw=1.4, zorder=4, label="Human (E2 vs E1)")
    ax.set_xticks(x, pcs); ax.set_xlabel("Held-out project"); ax.set_ylabel(yl); grid_y(ax)
ax = axes[2]; panel(ax, "c", x=-0.3)
ax.scatter(x - 0.12, obs_gap.loc[pcs], s=22, marker="o", color=INK, label="Observed E2 − E1")
ax.scatter(x, cv.loc[pcs, "gap"], s=20, marker="o", color=BLUE, edgecolor="white", lw=0.4, label="Rater heads, official CV")
ax.scatter(x + 0.12, lf.loc[pcs, "gap"], s=20, marker="s", color=ORANGE, edgecolor="white", lw=0.4, label="Rater heads, LOPO")
ax.axhline(0, color=AXIS, lw=0.8); ax.set_xticks(x, pcs); ax.set_xlabel("Project"); ax.set_ylabel("Mean FI gap (E2 − E1)"); grid_y(ax)
h, l = axes[0].get_legend_handles_labels(); h2, l2 = ax.get_legend_handles_labels()
fig.legend(h + h2, l + l2, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.24), fontsize=6.3, columnspacing=0.9)
save(fig, "fig6_lopo")

json.dump({"lopo_gap": lf.gap.round(4).to_dict(), "cv_gap": cv.gap.round(4).to_dict(), "obs_gap": obs_gap.round(4).to_dict(),
           "human_dice_fib": hum.round(3).to_dict()}, open(OUT / "fig_numbers.json", "w"), indent=1)
print("done")
