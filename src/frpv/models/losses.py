"""Losses for the different label models.

label_mode
  e1 / e2     : latent prediction trained on a single expert
  mean_soft   : latent prediction trained on the average of the two experts' one-hot labels
  dual_loss   : average of per-expert losses on the SAME output (official baseline formulation)
  cm_pixel    : latent -> per-pixel confusion matrix per rater (Zhang et al., NeurIPS 2020) + trace reg.
  offset      : (ours) rater logits = latent logits + centred, image-dependent logit offsets;
                nested head => every rater-specific prediction is hierarchy-consistent too.

All terms are computed only inside the ROI (outside it there is no ground truth).
Region Dice is computed on the *nested regions* (full wall = classes 1+2, fibrosis = class 2),
matching the evaluation targets.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .network import logprobs3


def _masked_ce(logp3, target, roi):
    """logp3 (B,3,H,W); target int (B,H,W) or soft (B,3,H,W)."""
    if target.dtype in (torch.int64, torch.int32, torch.uint8):
        nll = -logp3.gather(1, target.long().unsqueeze(1)).squeeze(1)
    else:
        nll = -(target * logp3).sum(1)
    return (nll * roi).sum() / roi.sum().clamp_min(1.0)


def _region_dice(p3, target, roi, w_fib=2.0, eps=1.0):
    """Batch soft Dice on nested regions. p3 probs (B,3,H,W); target int (B,H,W) or soft (B,3,H,W)."""
    if target.dtype in (torch.int64, torch.int32, torch.uint8):
        t3 = F.one_hot(target.long(), 3).permute(0, 3, 1, 2).float()
    else:
        t3 = target
    pw, pf = (p3[:, 1] + p3[:, 2]) * roi, p3[:, 2] * roi
    tw, tf = (t3[:, 1] + t3[:, 2]) * roi, t3[:, 2] * roi
    dw = 1 - (2 * (pw * tw).sum() + eps) / (pw.sum() + tw.sum() + eps)
    df = 1 - (2 * (pf * tf).sum() + eps) / (pf.sum() + tf.sum() + eps)
    return (dw + w_fib * df) / (1 + w_fib)


def _seg_loss(logp3, target, roi, lc):
    return lc.ce_weight * _masked_ce(logp3, target, roi) + \
        lc.dice_weight * _region_dice(logp3.exp(), target, roi, lc.dice_fib_weight)


def _targets_wf(target):
    if target.dtype in (torch.int64, torch.int32, torch.uint8):
        return (target > 0).float(), (target == 2).float()
    return target[:, 1] + target[:, 2], target[:, 2]


def _multilabel_loss(z, target, roi, lc, eps=1.0):
    """Two independent sigmoids (wall, fibrosis) - ablation without the hierarchy."""
    tw, tf = _targets_wf(target)
    z = z.float()
    bce = (F.binary_cross_entropy_with_logits(z[:, 0], tw, reduction="none") +
           F.binary_cross_entropy_with_logits(z[:, 1], tf, reduction="none"))
    ce = (bce * roi).sum() / roi.sum().clamp_min(1.0)
    pw, pf = torch.sigmoid(z[:, 0]) * roi, torch.sigmoid(z[:, 1]) * roi
    dw = 1 - (2 * (pw * tw * roi).sum() + eps) / (pw.sum() + (tw * roi).sum() + eps)
    df = 1 - (2 * (pf * tf * roi).sum() + eps) / (pf.sum() + (tf * roi).sum() + eps)
    w = lc.dice_fib_weight
    return lc.ce_weight * ce + lc.dice_weight * (dw + w * df) / (1 + w)


def seg_loss_logits(z, target, roi, head, lc):
    if head == "multilabel":
        return _multilabel_loss(z, target, roi, lc)
    return _seg_loss(logprobs3(z, head), target, roi, lc)


def _tv(x):
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def compute_loss(out: dict, y: torch.Tensor, roi: torch.Tensor, cfg, model=None):
    """y: (B,R,H,W) categorical labels of each rater; roi (B,H,W) float."""
    head, mode, lc = cfg.model.head, cfg.model.label_mode, cfg.loss
    z = out["z"]
    logs = {}
    if mode in ("e1", "e2"):
        loss = seg_loss_logits(z, y[:, int(mode[1]) - 1], roi, head, lc)
    elif mode == "mean_soft":
        t = F.one_hot(y.long(), 3).float().mean(1).permute(0, 3, 1, 2)   # (B,3,H,W)
        loss = seg_loss_logits(z, t, roi, head, lc)
    elif mode == "dual_loss":
        loss = sum(seg_loss_logits(z, y[:, r], roi, head, lc) for r in range(y.shape[1])) / y.shape[1]
    elif mode == "offset":
        d = out["delta"]
        loss = 0.0
        for r in range(y.shape[1]):
            loss = loss + seg_loss_logits(z + d[:, r], y[:, r], roi, head, lc)
        loss = loss / y.shape[1]
        reg = lc.offset_l2 * (d.float() ** 2).mean() + lc.offset_tv * _tv(d.float())
        logs["reg_offset"] = float(reg.detach())
        loss = loss + reg
    elif mode == "cm_pixel":
        p = logprobs3(z, head).exp()
        cm = out["cm"].float()
        loss = 0.0
        for r in range(y.shape[1]):
            q = torch.einsum("bihw,bijhw->bjhw", p, cm[:, r]).clamp_min(1e-6)
            loss = loss + _seg_loss(q.log(), y[:, r], roi, lc)
        loss = loss / y.shape[1]
        tr = cm.diagonal(dim1=2, dim2=3).sum(-1)                       # (B,R,H,W)
        reg = lc.cm_trace * ((tr * roi.unsqueeze(1)).sum() / roi.sum().clamp_min(1) / y.shape[1])
        logs["reg_trace"] = float(reg.detach())
        loss = loss + reg
    else:
        raise ValueError(mode)
    if model is not None and getattr(model, "stain", None) is not None and cfg.stain.learnable:
        loss = loss + cfg.stain.matrix_reg * model.stain.regularizer()
    logs["loss"] = float(loss.detach())
    return loss, logs
