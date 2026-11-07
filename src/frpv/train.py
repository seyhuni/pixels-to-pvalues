"""Training of one fold + out-of-fold inference.

results/<exp_name>/
    config.yaml
    fold<k>/train_log.csv, inner_val.csv, model_last.pt, model_best.pt, predictions.csv
    oof_predictions.csv          (concatenated over folds by scripts/03_collect.py or run_cv)
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import resolve_path, save_config
from .data.dataset import EpochSampler, TrainCrops, worker_init
from .data.meta import fold_column, inner_val_animals, load_meta
from .eval.infer import run_inference
from .models.losses import compute_loss
from .models.network import FibrosisNet
from .stain.physics import estimate_from_cache, load_stain_matrix


def seed_all(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stain_payload(cfg, path=None):
    """(stain matrix, concentration stats). `path` = per-fold json written by run_fold; falls back to
    the global file from 01_prepare_data.py (descriptive only)."""
    path = Path(path) if path is not None else resolve_path(cfg.stain.vectors_file)
    m = load_stain_matrix(path)
    stats = None
    if path.exists():
        d = json.loads(path.read_text())
        stats = d.get("concentration_stats")
    return m, stats


def build_model(cfg, device, stain_file=None):
    m, stats = stain_payload(cfg, stain_file)
    model = FibrosisNet(cfg, stain_matrix=m, conc_stats=stats).to(device)
    return model


def trainable_state(model) -> dict:
    """Only trainable parameters + buffers (frozen foundation-model weights are re-loaded from the
    hub at load time) -> checkpoints of a few MB-100 MB instead of several GB."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    keep |= {n for n, _ in model.named_buffers()}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}


def load_trainable(model, sd):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    bad = [k for k in missing if k in trainable]
    if bad:
        raise RuntimeError(f"checkpoint lacks trainable weights: {bad[:5]}")
    return model


def exp_dir(cfg) -> Path:
    return resolve_path(cfg.paths.results_dir) / cfg.exp_name


def _cosine(epoch_f, total, warm):
    if epoch_f < warm:
        return max(epoch_f / max(warm, 1e-8), 1e-3)
    t = (epoch_f - warm) / max(total - warm, 1e-8)
    return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def load_checkpoint(cfg, path, device):
    path = Path(path)
    sf = path.parent / "stain_vectors.json"
    model = build_model(cfg, device, sf if sf.exists() else None)
    sd = torch.load(path, map_location=device, weights_only=True)
    load_trainable(model, sd)
    model.eval()
    return model


def run_fold(cfg, fold: int, skip_if_done=True, progress=True) -> pd.DataFrame | None:
    out_dir = exp_dir(cfg) / f"fold{fold}"
    pred_csv = out_dir / "predictions.csv"
    if skip_if_done and pred_csv.exists():
        print(f"[skip] {pred_csv} exists")
        return pd.read_csv(pred_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, exp_dir(cfg) / "config.yaml")
    seed_all(cfg.train.seed + 1000 * fold, cfg.train.deterministic)
    device = get_device()
    cache = resolve_path(cfg.paths.cache_dir)
    meta = load_meta(cache)
    fcol = fold_column(cfg.data.split)
    test_meta = meta[meta[fcol] == fold]
    train_all = meta[meta[fcol] != fold]
    val_an = inner_val_animals(train_all["animal"].unique().tolist(), cfg.data.inner_val_frac,
                               seed=cfg.train.seed + fold)
    val_meta = train_all[train_all["animal"].isin(val_an)]
    train_meta = train_all[~train_all["animal"].isin(val_an)]
    dbg = cfg.get("debug_max_images")
    if dbg:                                   # smoke tests only
        test_meta, val_meta = test_meta.head(int(dbg)), val_meta.head(int(dbg))
    print(f"fold {fold}: train {train_meta.animal.nunique()} animals/{len(train_meta)} imgs | "
          f"inner-val {len(val_an)}/{len(val_meta)} | test {test_meta.animal.nunique()}/{len(test_meta)}")
    (out_dir / "split.json").write_text(json.dumps({
        "train_animals": sorted(train_meta.animal.unique().tolist()), "val_animals": val_an,
        "test_animals": sorted(test_meta.animal.unique().tolist())}, indent=1))

    # stain vectors / concentration statistics from the TRAINING animals of this fold only
    sf = out_dir / "stain_vectors.json"
    if not sf.exists():
        m_fold, st_fold = estimate_from_cache(train_meta.stem.tolist(), cache, seed=fold)
        sf.write_text(json.dumps({"global": m_fold.tolist(), "concentration_stats": st_fold}, indent=1))
    m_stain, _ = stain_payload(cfg, sf)
    ds = TrainCrops(train_meta, cache, cfg.data.crop_size, cfg.data.samples_per_epoch,
                    cfg.data.scale_jitter, cfg.data.roi_center_prob, cfg.stain.augment, m_stain,
                    dict(conc_scale=cfg.stain.aug_conc_scale, conc_shift=cfg.stain.aug_conc_shift,
                         vector_deg=cfg.stain.aug_vector_deg),
                    cfg.data.sample_weighting, seed=cfg.train.seed + fold)
    nw = int(cfg.train.num_workers)
    sampler = EpochSampler(len(ds))
    pin = bool(cfg.train.get("pin_memory", False)) and device.type == "cuda"
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, sampler=sampler, num_workers=nw,
                    pin_memory=pin, drop_last=True, persistent_workers=nw > 0,
                    worker_init_fn=worker_init if nw > 0 else None)
    model = build_model(cfg, device, sf)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"model params: trainable {n_train/1e6:.2f}M / total {n_all/1e6:.2f}M")
    opt = torch.optim.AdamW(model.param_groups(cfg.train.lr, cfg.train.lr_encoder, cfg.train.weight_decay))
    base_lrs = [g["lr"] for g in opt.param_groups]
    amp = cfg.train.amp if device.type == "cuda" else "none"
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))
    fwd = torch.compile(model) if cfg.train.compile else model

    steps_per_epoch = len(dl)
    E = int(cfg.train.epochs)
    log_rows, best_score, t0, start_epoch = [], -1e9, time.time(), 0
    resume_p = out_dir / "resume.pt"
    if (out_dir / "model_last.pt").exists():
        start_epoch = E                                   # training finished; only inference was missing
        print(f"[resume] {out_dir.name}: model_last.pt found - skipping training")
    elif resume_p.exists():
        st = torch.load(resume_p, map_location=device, weights_only=False)
        load_trainable(model, st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, best_score, log_rows = st["epoch"], st["best_score"], st["log_rows"]
        t0 = time.time() - 60 * st.get("elapsed_min", 0.0)
        print(f"[resume] continuing {out_dir.parent.name}/{out_dir.name} from epoch {start_epoch + 1}/{E}")
    for epoch in range(start_epoch, E):
        sampler.set_epoch(epoch)
        model.train()
        agg = {}
        try:
            batches = iter(dl)                  # (re)starts DataLoader worker processes
        except OSError as e:                    # e.g. WinError 4551: Windows App Control blocked the spawn
            print(f"[warn] DataLoader workers could not be started ({e}); continuing with num_workers=0")
            with open(out_dir / "warnings.log", "a", encoding="utf-8") as fh:
                fh.write(f"epoch {epoch + 1}: worker spawn failed -> num_workers=0 ({e})\n")
            dl = DataLoader(ds, batch_size=cfg.train.batch_size, sampler=sampler, num_workers=0,
                            pin_memory=pin, drop_last=True)
            batches = iter(dl)
        it = batches
        if progress:
            from tqdm import tqdm
            it = tqdm(batches, total=len(dl), desc=f"ep {epoch+1}/{E}", leave=False)
        for step, b in enumerate(it):
            f = _cosine(epoch + step / steps_per_epoch, E, cfg.train.warmup_epochs)
            for g, lr0 in zip(opt.param_groups, base_lrs):
                g["lr"] = lr0 * f
            x = b["image"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)
            roi = b["roi"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dt, enabled=dt is not None):
                out = fwd(x)
            loss, logs = compute_loss(out, y, roi, cfg, model)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
        row = {"epoch": epoch + 1, "time_min": (time.time() - t0) / 60,
               **{k: v / steps_per_epoch for k, v in agg.items()}}
        if len(val_meta) and ((epoch + 1) % cfg.train.val_every == 0 or epoch + 1 == E):
            model.eval()
            icfg = type(cfg.infer)(dict(cfg.infer, tta=False))
            vdf = run_inference([model], val_meta, cache, icfg, cfg.model.head, device, amp, progress=False)
            row.update({"val_dice_wall_e1": vdf.dice_wall_e1.mean(), "val_dice_fib_e1": vdf.dice_fib_e1.mean(),
                        "val_dice_fib_e2": vdf.dice_fib_e2.mean(),
                        "val_FI_mae_mean": (vdf.FI_hat - vdf.FI_mean).abs().mean()})
            score = 0.5 * (row["val_dice_fib_e1"] + row["val_dice_fib_e2"]) - row["val_FI_mae_mean"]
            if score > best_score and cfg.train.select == "best":
                best_score = score
                torch.save(trainable_state(model), out_dir / "model_best.pt")
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        # crash-safe resume point (atomic replace) - a crash costs at most one epoch
        tmp = out_dir / "resume.tmp"
        torch.save({"model": trainable_state(model), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "best_score": best_score, "log_rows": log_rows,
                    "elapsed_min": (time.time() - t0) / 60}, tmp)
        os.replace(tmp, resume_p)
    trained_now = start_epoch < E
    if trained_now:
        torch.save(trainable_state(model), out_dir / "model_last.pt")
        if hasattr(model, "stain") and model.stain is not None:
            (out_dir / "stain_matrix_learned.json").write_text(
                json.dumps(model.stain.matrix().detach().cpu().tolist()))
    if resume_p.exists():
        resume_p.unlink()

    # ---- out-of-fold inference on the test fold
    ck = out_dir / ("model_best.pt" if cfg.train.select == "best" and (out_dir / "model_best.pt").exists()
                    else "model_last.pt")
    load_trainable(model, torch.load(ck, map_location=device, weights_only=True))
    model.eval()
    pm = (out_dir / "probmaps") if cfg.infer.save_probmaps else None
    df = run_inference([model], test_meta, cache, cfg.infer, cfg.model.head, device, amp, pm, progress)
    df.insert(0, "fold", fold)
    df.insert(1, "exp", cfg.exp_name)
    df.to_csv(pred_csv, index=False)
    print(f"fold {fold} test: dice_wall_e1={df.dice_wall_e1.mean():.3f} dice_fib_e1={df.dice_fib_e1.mean():.3f} "
          f"dice_fib_e2={df.dice_fib_e2.mean():.3f} FI_MAE(mean)={(df.FI_hat-df.FI_mean).abs().mean():.4f}")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return df


def collect_oof(cfg) -> pd.DataFrame | None:
    d = exp_dir(cfg)
    parts = [pd.read_csv(p) for p in sorted(d.glob("fold*/predictions.csv"))]
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(d / "oof_predictions.csv", index=False)
    return df
