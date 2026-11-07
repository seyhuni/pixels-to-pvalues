"""Full network: context encoder (CNN or pathology FM) + stain-physics detail branch +
fusion decoder + nested / flat head + rater-specific components.

Outputs (dict)
  z      latent ("consensus") logits         (B,K,H,W)   K=2 nested, K=3 flat
  delta  rater logit offsets (centred)       (B,R,K,H,W) label_mode == 'offset'
  cm     rater confusion matrices (row=true) (B,R,3,3,H,W) label_mode == 'cm_pixel'
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import build_encoder
from ..stain.physics import StainDeconv


def conv_bn(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.GELU())


class ConvBlock(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(conv_bn(i, o), conv_bn(o, o))


class CNNContextDecoder(nn.Module):
    """UNet-style top-down decoder over CNN pyramid -> stride-4 feature map."""

    def __init__(self, channels, dim):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in channels])
        self.fuse = nn.ModuleList([ConvBlock(2 * dim, dim) for _ in channels[:-1]])

    def forward(self, feats):
        x = self.lat[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            skip = self.lat[i](feats[i])
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.fuse[i](torch.cat([x, skip], 1))
        return x


class ViTContextDecoder(nn.Module):
    """DPT-like reassembly of 4 ViT layers into a pyramid (x4, x2, x1, /2) + FPN fusion."""

    def __init__(self, embed_dim, dim):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(embed_dim, dim, 1) for _ in range(4)])
        self.resample = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(dim, dim, 2, 2), nn.GELU(), nn.ConvTranspose2d(dim, dim, 2, 2)),
            nn.ConvTranspose2d(dim, dim, 2, 2),
            nn.Identity(),
            nn.Conv2d(dim, dim, 3, 2, 1),
        ])
        self.fuse = nn.ModuleList([ConvBlock(2 * dim, dim) for _ in range(3)])

    def forward(self, maps):
        p = [self.resample[i](self.proj[i](m)) for i, m in enumerate(maps)]
        x = p[3]
        for i in (2, 1, 0):
            x = F.interpolate(x, size=p[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.fuse[i](torch.cat([x, p[i]], 1))
        return x


class DetailBranch(nn.Module):
    """Light high-resolution CNN on stain concentrations and/or RGB (native 0.252 um/px)."""

    def __init__(self, in_ch, chans=(32, 64, 96, 128)):
        super().__init__()
        c1, c2, c4, c8 = chans
        self.s1 = ConvBlock(in_ch, c1)
        self.s2 = nn.Sequential(conv_bn(c1, c2, s=2), conv_bn(c2, c2))
        self.s4 = nn.Sequential(conv_bn(c2, c4, s=2), conv_bn(c4, c4))
        self.s8 = nn.Sequential(conv_bn(c4, c8, s=2), conv_bn(c8, c8))

    def forward(self, x):
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f4 = self.s4(f2)
        f8 = self.s8(f4)
        return {1: f1, 2: f2, 4: f4, 8: f8}


class FusionDecoder(nn.Module):
    def __init__(self, dim, detail_chans, start_stride, use_detail=True):
        super().__init__()
        self.strides = [s for s in (8, 4, 2, 1) if s <= start_stride]
        dch = dict(zip((1, 2, 4, 8), detail_chans))
        widths = {8: dim, 4: max(dim // 2, 64), 2: max(dim // 4, 48), 1: max(dim // 8, 32)}
        self.use_detail = use_detail
        blocks, prev = [], dim
        for s in self.strides:
            inc = prev + (dch[s] if use_detail else 0)
            blocks.append(ConvBlock(inc, widths[s]))
            prev = widths[s]
        self.blocks = nn.ModuleList(blocks)
        self.out_ch = prev

    def forward(self, ctx, detail, H, W):
        x = ctx
        for s, blk in zip(self.strides, self.blocks):
            size = (-(-H // s), -(-W // s)) if detail is None else detail[s].shape[-2:]
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            if self.use_detail:
                x = torch.cat([x, detail[s]], 1)
            x = blk(x)
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x


class FibrosisNet(nn.Module):
    def __init__(self, cfg, stain_matrix=None, conc_stats=None):
        super().__init__()
        m, sc = cfg.model, cfg.stain
        self.head_type = m.head
        self.label_mode = m.label_mode
        self.R = int(m.n_raters)
        self.K = 3 if m.head == "flat" else 2
        if m.head == "multilabel" and m.label_mode == "cm_pixel":
            raise ValueError("cm_pixel needs a 3-class head (nested or flat)")
        self.encoder = build_encoder(m)
        dim = m.decoder_dim
        if self.encoder.kind == "cnn":
            self.ctx = CNNContextDecoder(self.encoder.channels, dim)
            start = 4
        else:
            self.ctx = ViTContextDecoder(self.encoder.embed_dim, dim)
            start = 8
        self.branch_input = sc.branch_input
        use_detail = sc.branch_input != "none"
        if "conc" in sc.branch_input:
            cm, cs = (conc_stats or {}).get("mean"), (conc_stats or {}).get("std")
            self.stain = StainDeconv(stain_matrix, learnable=bool(sc.learnable), conc_mean=cm, conc_std=cs)
        else:
            self.stain = None
        in_ch = {"conc": 3, "rgb": 3, "rgb+conc": 6, "none": 0}[sc.branch_input]
        self.detail = DetailBranch(in_ch, m.detail_channels) if use_detail else None
        self.fusion = FusionDecoder(dim, m.detail_channels, start, use_detail)
        c = self.fusion.out_ch
        self.head = nn.Conv2d(c, self.K, 1)
        self.offset_head = None
        self.cm_head = None
        if self.label_mode == "offset":
            n_off = self.K if self.R == 2 else self.R * self.K
            self.offset_head = nn.Sequential(ConvBlock(c, c), nn.Conv2d(c, n_off, 1))
            nn.init.zeros_(self.offset_head[-1].weight)
            nn.init.zeros_(self.offset_head[-1].bias)
        elif self.label_mode == "cm_pixel":
            self.cm_head = nn.Sequential(ConvBlock(c, c), nn.Conv2d(c, self.R * 9, 1))
            nn.init.zeros_(self.cm_head[-1].weight)
            with torch.no_grad():
                eye = torch.eye(3).flatten() * 4.0          # start close to identity
                self.cm_head[-1].bias.copy_(eye.repeat(self.R))

    def _detail_input(self, x01):
        if self.branch_input == "rgb":
            return (x01 - 0.5) / 0.25
        if self.branch_input == "conc":
            return self.stain(x01)
        if self.branch_input == "rgb+conc":
            return torch.cat([(x01 - 0.5) / 0.25, self.stain(x01)], 1)
        return None

    def forward(self, x01):
        H, W = x01.shape[-2:]
        feats = self.encoder(x01)
        ctx = self.ctx(feats)
        det = self.detail(self._detail_input(x01)) if self.detail is not None else None
        f = self.fusion(ctx, det, H, W)
        out = {"z": self.head(f)}
        if self.offset_head is not None:
            d = self.offset_head(f)
            if self.R == 2:
                d = torch.stack([d, -d], 1)                     # sum-to-zero by construction
            else:
                d = d.view(d.shape[0], self.R, self.K, H, W)
                d = d - d.mean(1, keepdim=True)
            out["delta"] = d
        if self.cm_head is not None:
            cm = self.cm_head(f).view(f.shape[0], self.R, 3, 3, H, W)
            out["cm"] = torch.softmax(cm, dim=3)                 # rows: true class, cols: observed
        return out

    def param_groups(self, lr, lr_enc, wd):
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        enc_ids = {id(p) for p in enc}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in enc_ids]
        groups = [{"params": rest, "lr": float(lr), "weight_decay": float(wd)}]
        if enc:
            groups.append({"params": enc, "lr": float(lr_enc), "weight_decay": float(wd)})
        return groups


# ----------------------------------------------------------------------------- probability algebra
def logprobs3(z: torch.Tensor, head: str) -> torch.Tensor:
    """3-class log-probabilities (bg, wall-only, fibrosis) from head logits; dim=1."""
    if head == "flat":
        return F.log_softmax(z.float(), 1)
    if head == "multilabel":
        raise ValueError("multilabel head has no 3-class distribution; use wall_fib_probs")
    z = z.float()
    lw, lnw = F.logsigmoid(z[:, 0]), F.logsigmoid(-z[:, 0])
    lf, lnf = F.logsigmoid(z[:, 1]), F.logsigmoid(-z[:, 1])
    return torch.stack([lnw, lw + lnf, lw + lf], 1)


def probs3(z, head):
    return logprobs3(z, head).exp()


def wall_fib_probs(z: torch.Tensor, head: str):
    """(p_wall, p_fib) maps (B,H,W). For nested/flat p_fib <= p_wall pointwise (soft containment);
    for the 'multilabel' ablation (two independent sigmoids) containment is NOT guaranteed."""
    z = z.float()
    if head == "multilabel":
        return torch.sigmoid(z[:, 0]), torch.sigmoid(z[:, 1])
    p = probs3(z, head)
    return p[:, 1] + p[:, 2], p[:, 2]


def rater_wall_fib(out: dict, head: str, label_mode: str):
    """List of R (p_wall, p_fib) tuples with each rater's predicted maps, or None."""
    if label_mode == "offset" and "delta" in out:
        return [wall_fib_probs(out["z"] + out["delta"][:, r], head) for r in range(out["delta"].shape[1])]
    if label_mode == "cm_pixel" and "cm" in out:
        p = probs3(out["z"], head)                               # (B,3,H,W)
        res = []
        for r in range(out["cm"].shape[1]):
            q = torch.einsum("bihw,bijhw->bjhw", p, out["cm"][:, r].float())
            res.append((q[:, 1] + q[:, 2], q[:, 2]))
        return res
    return None
