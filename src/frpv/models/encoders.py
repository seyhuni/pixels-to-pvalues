"""Encoders: ImageNet CNNs (timm) and pathology foundation models (ViTs).

Spec strings
  cnn:<timm_model_name>          e.g. cnn:convnext_small.fb_in22k, cnn:resnet50.a1_in1k
  fm:<key>                       key in FM_REGISTRY below

Gated HuggingFace models (UNI, UNI2-h, Virchow2, H-optimus-*) require that you accepted the
licence on huggingface.co and ran `huggingface-cli login` (or set HF_TOKEN) on the workstation.
'dinov2_l' / 'dinov2_s' are open (natural-image DINOv2) and serve as non-pathology FM controls.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import apply_lora

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
HOPT_MEAN, HOPT_STD = (0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)


def _swiglu_kwargs():
    import timm
    return dict(mlp_layer=timm.layers.SwiGLUPacked, act_layer=nn.SiLU)


FM_REGISTRY = {
    # key: (source, id, kwargs-factory, mean, std)
    "uni2h": ("timm", "hf-hub:MahmoodLab/UNI2-h",
              lambda: dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
                           embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
                           reg_tokens=8, dynamic_img_size=True, **_swiglu_kwargs()),
              IMNET_MEAN, IMNET_STD),
    "uni": ("timm", "hf-hub:MahmoodLab/uni",
            lambda: dict(init_values=1e-5, dynamic_img_size=True, num_classes=0), IMNET_MEAN, IMNET_STD),
    "virchow2": ("timm", "hf-hub:paige-ai/Virchow2",
                 lambda: dict(dynamic_img_size=True, num_classes=0, **_swiglu_kwargs()), IMNET_MEAN, IMNET_STD),
    "hoptimus0": ("timm", "hf-hub:bioptimus/H-optimus-0",
                  lambda: dict(init_values=1e-5, dynamic_img_size=True, num_classes=0), HOPT_MEAN, HOPT_STD),
    "hoptimus1": ("timm", "hf-hub:bioptimus/H-optimus-1",
                  lambda: dict(init_values=1e-5, dynamic_img_size=True, num_classes=0), HOPT_MEAN, HOPT_STD),
    "dinov2_l": ("timm", "vit_large_patch14_dinov2.lvd142m",
                 lambda: dict(dynamic_img_size=True, num_classes=0), IMNET_MEAN, IMNET_STD),
    "dinov2_s": ("timm", "vit_small_patch14_dinov2.lvd142m",
                 lambda: dict(dynamic_img_size=True, num_classes=0), IMNET_MEAN, IMNET_STD),
    "phikon2": ("hf", "owkin/phikon-v2", None, IMNET_MEAN, IMNET_STD),
    "hibou_l": ("hf", "histai/hibou-L", None, (0.7068, 0.5755, 0.722), (0.195, 0.2316, 0.1816)),
}


class CNNEncoder(nn.Module):
    kind = "cnn"

    def __init__(self, name: str, pretrained: bool = True, finetune: str = "full"):
        super().__init__()
        import timm
        self.net = timm.create_model(name, pretrained=pretrained, features_only=True)
        info = self.net.feature_info
        keep = [i for i, r in enumerate(info.reduction()) if r >= 4]
        self.keep = keep
        self.channels = [info.channels()[i] for i in keep]
        self.reductions = [info.reduction()[i] for i in keep]
        cfg = getattr(self.net, "pretrained_cfg", {}) or {}
        self.register_buffer("mean", torch.tensor(cfg.get("mean", IMNET_MEAN)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.get("std", IMNET_STD)).view(1, 3, 1, 1))
        if finetune == "frozen":
            for p in self.net.parameters():
                p.requires_grad = False

    def forward(self, x01):
        feats = self.net((x01 - self.mean) / self.std)
        return [feats[i] for i in self.keep]


class ViTEncoder(nn.Module):
    kind = "vit"

    def __init__(self, key: str, pretrained=True, finetune="lora", lora_rank=16, lora_alpha=32,
                 lora_dropout=0.0, last_k=4, grad_ckpt=True, layers=None, native_mpp=0.252,
                 fm_input_mpp=0.5):
        super().__init__()
        if key not in FM_REGISTRY:
            raise KeyError(f"Unknown FM '{key}'. Options: {list(FM_REGISTRY)}")
        src, ident, kw, mean, std = FM_REGISTRY[key]
        self.src = src
        if src == "timm":
            import timm
            self.net = timm.create_model(ident, pretrained=pretrained, **kw())
            self.patch = self.net.patch_embed.patch_size[0]
            self.embed_dim = self.net.embed_dim
            depth = len(self.net.blocks)
            if grad_ckpt:
                self.net.set_grad_checkpointing(True)
        else:
            from transformers import AutoModel
            self.net = AutoModel.from_pretrained(ident, trust_remote_code=True) if pretrained else None
            if self.net is None:
                raise ValueError("HF encoders need pretrained=true")
            c = self.net.config
            ps = getattr(c, "patch_size", 14)
            self.patch = ps if isinstance(ps, int) else ps[0]
            self.embed_dim = c.hidden_size
            depth = c.num_hidden_layers
            if grad_ckpt and hasattr(self.net, "gradient_checkpointing_enable"):
                try:   # non-reentrant: frozen embeddings would otherwise block LoRA gradients
                    self.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    try:
                        self.net.gradient_checkpointing_enable()
                    except Exception:
                        pass
                except Exception:
                    pass
        self.layers = list(layers) if layers else [int(round(i * depth / 4)) - 1 for i in range(1, 5)]
        self.scale = native_mpp / fm_input_mpp
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.channels = [self.embed_dim] * 4
        self.n_lora = 0
        if finetune == "lora":
            self.n_lora = apply_lora(self.net, lora_rank, lora_alpha, lora_dropout)
        elif finetune == "frozen":
            for p in self.net.parameters():
                p.requires_grad = False
        elif finetune == "last_k":
            for p in self.net.parameters():
                p.requires_grad = False
            blocks = self.net.blocks if src == "timm" else self.net.encoder.layer
            for b in list(blocks)[-last_k:]:
                for p in b.parameters():
                    p.requires_grad = True

    def fm_size(self, h, w):
        r = lambda v: max(self.patch, int(round(v * self.scale / self.patch)) * self.patch)
        return r(h), r(w)

    def forward(self, x01):
        h, w = x01.shape[-2:]
        x = (x01 - self.mean) / self.std
        th, tw = self.fm_size(h, w)
        if (th, tw) != (h, w):
            x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False,
                              antialias=th < h)
        if self.src == "timm":
            return self.net.forward_intermediates(x, indices=self.layers, output_fmt="NCHW",
                                                  intermediates_only=True)
        kwargs = dict(output_hidden_states=True)
        try:
            out = self.net(pixel_values=x, interpolate_pos_encoding=True, **kwargs)
        except TypeError:
            out = self.net(pixel_values=x, **kwargs)
        gh, gw = th // self.patch, tw // self.patch
        maps = []
        for li in self.layers:
            t = out.hidden_states[li + 1]
            t = t[:, t.shape[1] - gh * gw:, :]
            maps.append(t.transpose(1, 2).reshape(t.shape[0], -1, gh, gw))
        return maps


def build_encoder(mcfg) -> nn.Module:
    spec = mcfg.encoder
    kind, name = spec.split(":", 1)
    if kind == "cnn":
        return CNNEncoder(name, pretrained=mcfg.pretrained,
                          finetune="frozen" if mcfg.finetune == "frozen" else "full")
    if kind == "fm":
        return ViTEncoder(name, pretrained=mcfg.pretrained, finetune=mcfg.finetune,
                          lora_rank=mcfg.lora_rank, lora_alpha=mcfg.lora_alpha,
                          lora_dropout=mcfg.lora_dropout, last_k=mcfg.last_k,
                          grad_ckpt=mcfg.grad_checkpointing, layers=mcfg.fm_layers,
                          native_mpp=mcfg.native_mpp, fm_input_mpp=mcfg.fm_input_mpp)
    raise ValueError(f"Bad encoder spec {spec}")
