# From pixels to p-values

Code for the paper *From pixels to p-values: rater-aware, hierarchy-consistent and stain-physics-informed
segmentation with distribution-free guarantees for histological fibrosis endpoints* (Hüseyin Yanık, submitted to
*Machine Learning: Science and Technology*).

The study uses the public **FRPV-PH** dataset (Karpov *et al* 2026, *Scientific Data*,
[doi:10.1038/s41597-026-08214-y](https://doi.org/10.1038/s41597-026-08214-y)). It has 705 Picro-Mallory
micrographs of rat pulmonary arteries from 68 animals, each annotated independently by two experts (vascular wall
and intramural fibrosis).

## What is in here

| Component | Where |
|---|---|
| Nested probabilistic head: p(wall) · p(fibrosis \| wall), so fibrosis ⊆ wall holds by construction | `src/frpv/models/network.py`, `losses.py` |
| Zero-sum rater logit offsets (consensus + expert-specific outputs), plus the mean-soft, dual-loss, CM-pixel and multilabel baselines | `src/frpv/models/network.py`, `losses.py` |
| Learnable three-dye Beer–Lambert deconvolution layer (aniline blue, acid fuchsin, orange G), physically consistent stain augmentation and a native-resolution detail branch | `src/frpv/stain/physics.py` |
| Pathology foundation-model encoders (UNI2-h, Virchow2, DINOv2-L) with LoRA, DPT-style decoder, ConvNeXt-S CNN | `src/frpv/models/encoders.py`, `lora.py` |
| Sliding-window inference with test-time augmentation, metrics (Dice, MSD, GED, containment, FI MAE/CCC/ICC) | `src/frpv/eval/` |
| Split-conformal intervals for the fibrosis index (FI): naive, hierarchical pooled, subsample, animal level, Mondrian; five difficulty scales | `src/frpv/uq/conformal.py` |
| Prediction-powered inference (PPI and PPI++) for between-group contrasts with animals as units | `src/frpv/stats/ppi.py` |
| Linear mixed models, animal bootstrap, rater-dependence analysis | `src/frpv/stats/analysis.py`, `scripts/07_rater_analysis.py` |
| Baselines: classical thresholding, re-run of the published UNet (as released and with corrected evaluation), nnU-Net 2D | `baselines/` |

## Main results (official animal-level 5-fold CV, out-of-fold, 705 images)

| | Wall Dice | Fibrosis Dice | FI MAE | FI CCC vs E1 |
|---|---|---|---|---|
| Expert E2 vs expert E1 | 0.955 | 0.609 | 0.050 | 0.53 |
| Proposed, UNI2-h, 3-seed ensemble | 0.963 | 0.663 | 0.029 | 0.69 |

- **Conformal intervals.** With rater-gap scaling, worst-project coverage of animal-level FI rises from 0.60 to
  0.89 at a nominal 0.90.
- **Prediction-powered inference.** A naive model-only analysis underestimates the CTEPH effect in project M by
  about one fifth. PPI removes this bias and keeps the type-I error at about 0.05.

The full tables are in the paper.

## Installation

```bat
conda env create -f environment.yml
conda activate mlst
pip install -r requirements-baselines.txt   :: optional: MONAI + nnU-Net baselines
```

The environment uses PyTorch 2.7.1 with CUDA 12.8 wheels, which also support RTX 50xx GPUs. For an existing
environment, see the comments in `requirements.txt`.

**Gated foundation models.** UNI2-h (`MahmoodLab/UNI2-h`) and Virchow2 (`paige-ai/Virchow2`) require accepting
their licences on Hugging Face and then running `huggingface-cli login`. DINOv2-L and ConvNeXt-S are open.

## Data

1. Download FRPV-PH from [figshare](https://doi.org/10.6084/m9.figshare.31386748) or the
   [dataset repository](https://github.com/Sinitca-Aleksandr/FRPV-PH).
2. Extract it to `data/FRPV-PH/`, so that it contains `raw_images/`, `roi_masks/`, `expert_1/` and `expert_2/`.
3. Copy the official split file `splits_5fold.csv` from the dataset repository to
   `data/splits_5fold_official.csv`.

The dataset and the split file are not redistributed here. Please cite the dataset paper when you use them.

## Running

All settings are in `configs/base.yaml`. Experiment plans are in `plans/`. Every step resumes from where it
stopped.

```bat
RUN_1_setup_and_smoke.bat              :: environment check, data cache, unit tests, short GPU run
RUN_2_backbones.bat                    :: stage 1: backbone comparison
RUN_3_ablations_and_analysis.bat       :: stages 2-4 + baselines + conformal / PPI / rater analyses
RUN_4_complementary.bat                :: published-UNet re-run, LOPO stain-none, nnU-Net
```

On Linux, call the underlying scripts directly, for example:

```bash
python scripts/01_prepare_data.py
python scripts/run_plan.py plans/stage1_backbones.yaml
python scripts/02_train_cv.py --exp configs/exp/main.yaml model.encoder=fm:uni2h exp_name=test --folds 0
python scripts/05_conformal.py --exp final_ens_uni2h
python scripts/06_ppi.py --exp final_ens_uni2h
python scripts/09_make_figures.py results cache/meta.csv data/FRPV-PH figures_out
```

Results are written to `results/<experiment>/`:

- out-of-fold predictions for every image;
- summary tables;
- conformal grids;
- PPI simulations.

Checkpoints contain only the trained parameters (LoRA adapters, decoder and heads). Frozen foundation-model
weights are loaded from the Hugging Face cache.

## Hardware and run time

All experiments were run on a single NVIDIA RTX 5060 Ti (16 GB) with an Intel Core Ultra 9 CPU and 128 GB RAM.
Batch size was 4 with 640 × 640 crops and bf16. Run time per fold:

| Backbone | Minutes per fold |
|---|---|
| ConvNeXt-S | ~110 |
| DINOv2-L | ~135 |
| UNI2-h | ~195 |
| Virchow2 | ~208 |

## Citation

If you use this code, please cite the paper (citation details will be added on publication) and the dataset:

```bibtex
@article{Karpov2026,
  title   = {Microscopy Image Dataset of pulmonary vessels for Quantitative assessment of fibrosis},
  author  = {Karpov, Andrei A. and Sinitca, Aleksandr and Isakova, Nadezda P. and Dushkova, Anna S. and
             Semenova, Ekaterina V. and Karpenko, Vladislava V. and Smirnov, Sergey S. and Karpova, Yuliya I. and
             Vaulina, Dariya D. and Galagudza, Michael M. and Krasichkov, Alexander and Kaplun, Dmitrii},
  journal = {Scientific Data},
  year    = {2026},
  doi     = {10.1038/s41597-026-08214-y}
}
```

## Licence

The code is released under the MIT licence (see `LICENSE`). The FRPV-PH dataset and the foundation-model weights
keep their own licences.

## Acknowledgements

The author thanks the creators of FRPV-PH for making the images, both expert annotations and their baseline code
publicly available. A Turkish working guide for the run scripts is in `docs/README_TR.md`.
