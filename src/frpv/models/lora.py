"""Minimal, dependency-free LoRA for nn.Linear layers (works for timm and HF ViTs)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    @property
    def weight(self):  # some code paths (e.g. fused attention) read .weight
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(self.drop(x))) * self.scale


DEFAULT_TARGETS = ("attn.qkv", "attn.proj", "attention.query", "attention.key", "attention.value",
                   "attention.output.dense")


def apply_lora(model: nn.Module, rank=16, alpha=32, dropout=0.0, targets=DEFAULT_TARGETS) -> int:
    """Freeze `model` and inject LoRA into Linear layers whose qualified name ends with a target."""
    for p in model.parameters():
        p.requires_grad = False
    n = 0
    for name, module in list(model.named_modules()):
        for cname, child in list(module.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and any(full.endswith(t) for t in targets):
                setattr(module, cname, LoRALinear(child, rank, alpha, dropout))
                n += 1
    return n
