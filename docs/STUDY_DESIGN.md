# Study design (pre-specified analysis plan)

**Working title.** *From pixels to p-values: rater-aware, hierarchy-consistent and stain-physics-informed
segmentation with distribution-free guarantees for histological fibrosis endpoints*

**Target journal.** Machine Learning: Science and Technology (IOP), Research Paper.

**Dataset.** FRPV-PH (Karpov et al., *Sci. Data* 2026; figshare 10.6084/m9.figshare.31386748).
705 Picro-Mallory micrographs of rat pulmonary-artery segments (0.252 µm/px), 68 animals, 4 projects
(AH = monocrotaline PAH, M/RE/RA = CTEPH), groups healthy / untreated / treated; ROI + two independent
expert masks of the vascular wall and of intramural fibrosis (fibrosis ⊂ wall).

## 1. Motivation (from our own analysis of the released data)

| Observation | Value |
|---|---|
| Inter-rater Dice wall / fibrosis | 0.955 / 0.61 |
| Rater bias is project-dependent (E2/E1 fibrosis area) | M 1.83, RA 1.16, RE 0.78, AH 0.70 (wall ≈ 1.0) |
| Animal-level E1–E2 FI correlation | 0.61 |
| Within-animal share of FI variance | ≈ 48 % |
| Contrast conclusions that depend on the rater (LMM, α = 0.05) | AH1→AH3 p = 0.049 (E1) vs 0.34 (E2); RE1→RE3 p = 0.050 vs 0.16; M effect 0.089 vs 0.177 |
| Published baseline | thresholds raw logits (no softmax), anisotropic 512² resize, wall-only vs full-wall Dice → FI predicted ≈ 0.63 vs 0.13 true |
| Aniline-blue concentration alone separates fibrotic vs non-fibrotic wall pixels | AUC ≈ 0.91 (unsupervised 3-dye deconvolution) |

**Problem statement.** The scientific endpoint (FI and treatment effects on FI) is a *ratio of nested
areas* measured under (i) strong, context-dependent rater disagreement, (ii) clustered data (segments in
animals), (iii) staining-batch shift between projects. Pixel metrics (Dice) do not tell a scientist
whether an automated FI can replace manual FI for inference.

## 2. Contributions

1. **Hierarchy-consistent probabilistic head.** p(wall)·p(fib | wall); soft containment p_fib ≤ p_wall at
   every pixel and every threshold; decomposes errors into *boundary* (wall) and *criterion* (fib | wall).
2. **Rater-aware learning with centred, image-dependent logit offsets** (δ_1 = −δ_2 by construction →
   identifiable latent "consensus"; rater predictions stay hierarchy-consistent). Compared with
   single-rater, soft-label, dual-loss and per-pixel confusion-matrix (Zhang et al. 2020) training.
   The predicted rater gap is a learned estimate of *where experts would disagree*.
3. **Stain physics (Beer–Lambert) for the three-dye trichrome**: unsupervised 3-dye OD-vector
   estimation, physically consistent concentration-space augmentation, and a learnable deconvolution
   layer feeding a native-resolution detail branch, combined with a pathology foundation model
   operating at its pre-training resolution (0.5 µm/px).
4. **Endpoint-level distribution-free uncertainty**: hierarchical (animal-level) conformal intervals
   for the FI ratio on the logit scale with rater-aware adaptive scales; Mondrian per-project calibration.
5. **Valid downstream inference**: prediction-powered inference (PPI / PPI++) adapted to clustered,
   multi-arm preclinical designs; label-budget curves ("how many manually measured segments per animal").
6. **Corrected, reproducible benchmark** (official folds + leave-one-project-out) incl. nnU-Net and a
   re-implementation of the published baseline evaluated correctly.

## 3. Hypotheses

* **H1** Pathology FMs + stain-physics detail branch outperform ImageNet CNNs and nnU-Net on fibrosis
  Dice and FI agreement; the gap widens under leave-one-project-out.
* **H2** Rater-aware training (offset) ≥ dual-loss / soft-label on latent accuracy, and reproduces each
  expert's style (r-head closer to its own expert) and the project-specific bias direction.
* **H3** Nested head ≥ flat ≥ multilabel; multilabel produces containment violations.
* **H4** Beer–Lambert augmentation + concentration branch improves LOPO robustness vs HSV / none.
* **H5** Image-exchangeable conformal under-covers at animal level / per project; hierarchical +
  Mondrian calibration restores nominal coverage; rater-aware scales give shorter intervals.
* **H6** PPI keeps nominal coverage/type-I error and shrinks CIs relative to labelled-only analysis;
  naive use of predictions is biased in at least one contrast.

## 4. Experiments (plans/)

| Stage | Plan | Content |
|---|---|---|
| 0 | stage0_smoke | 2-epoch pipeline check (CNN + DINOv2-L) |
| 1 | stage1_backbones | full method × {ConvNeXt-S, DINOv2-L, UNI2-h, Virchow2, H-optimus-1}, official 5-fold |
| 2 | stage2_ablation | best backbone: label models (e1, e2, mean_soft, dual_loss, cm_pixel), heads (flat, multilabel), stain (none, hsv, BL-aug only, rgb+conc, fixed M, no detail branch), FM resolution |
| 3 | stage3_lopo | leave-one-project-out: full, mean_soft, dual_loss, no-stain |
| 4 | stage4_seeds | 2 extra seeds → 3-member deep ensemble (`final_ens`) |
| B | baselines | nnU-Net v2 (per expert), published UNet (as published vs corrected), colour-deconvolution threshold with oracle wall |

Training: 60 epochs × 2000 random 768² crops (ROI-centred p = 0.9, scale jitter with reflect padding,
D4 flips), AdamW, cosine LR, bf16. Inference: native resolution, Gaussian-blended sliding window,
4-flip TTA. Checkpoint: last epoch (no test-fold peeking). Backbone selection for stages 2–4 uses only
the inner-validation animals (15 % of the training animals of each fold), never the test folds.
Stain vectors and concentration statistics are re-estimated inside every fold from the training
animals only (no colour information from held-out animals/projects; matters for LOPO).
Checkpoints store only trainable weights (LoRA, decoder, heads, stain residual) + buffers.

## 5. Metrics

Segmentation (inside ROI, vs each expert): Dice, IoU, mean surface distance; GED for fibrosis
(rater-specific predictions vs both experts); containment violations.
Endpoint: FI MAE / bias / Lin's CCC / ICC(A,1) vs E1, E2 and their mean, image- and animal-level;
human reference = E2 evaluated against E1.
Uncertainty: animal-weighted coverage, coverage among finite intervals, worst-project coverage, mean
interval width, fraction of infinite intervals; two evaluation schemes: 500 random animal-level
calibration/test partitions (stratified by project) and leave-one-animal-out (maximal calibration
size, needed for Mondrian calibration in small projects). Scales are label-free (TTA / ensemble
spread, predicted rater gap), defined on the logit axis for the logit transform, with a soft floor.
Inference: (i) null calibration - two disjoint random animal sets from the same pool, type-I error;
(ii) real contrasts - arms subsampled without replacement (70 %), estimand = arm difference with all
expert labels (conservative, finite population); report coverage, CI width relative to the
labelled-only analysis, bias of the naive plug-in. PPI++ lambda = within-animal slope fitted
leave-one-animal-out.

## 6. Statistics

Animal-level percentile bootstrap (B = 2000) for all CIs; paired animal bootstrap for model
comparisons; random-intercept LMMs (animal) for group contrasts; no multiplicity correction for the
descriptive "rater dependence" table (reported as such), Holm correction for confirmatory model
comparisons (H1–H4) (`p_holm` in results/summary/paired_vs_<ref>.csv). GED compares sample sets
of different size for single-output vs rater-aware models - interpreted with that caveat.

## 7. Planned figures / tables

F1 graphical abstract · F2 data & rater-bias motivation (project-dependent bias, p-value flips) ·
F3 architecture · F4 qualitative results with rater heads and offset maps (boundary vs criterion) ·
F5 conformal coverage/width (naive vs pooled vs Mondrian; scale types) · F6 PPI coverage & label-budget
curves · T1 benchmark (official CV) · T2 LOPO · T3 ablations · T4 rater-style identifiability.

## 8. Threats to validity

Small number of animals per group (2–7) → wide CIs, emphasise estimation over testing; only two raters
(no ground truth) → targets E1, E2, mean and "both" are all reported; the same images were used to
define motivation and evaluation (motivating analysis uses expert labels only); gated FM licences
(non-commercial) — documented in the paper.
