"""LoRA (Low-Rank Adaptation) for timm Vision Transformers.

Injects trainable low-rank residuals into frozen ViT attention layers.
All original parameters are frozen; only lora_A and lora_B are trained.
"""
import math
from typing import Dict, List, Optional

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with LoRA side-path."""

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(linear.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, linear.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base + lora


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
) -> int:
    """Apply LoRA to targeted linear layers in-place. Returns count of adapted layers."""
    if target_modules is None:
        target_modules = ["attn.qkv", "attn.proj"]

    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in target_modules):
            continue
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_path, attr = parts
            parent = model
            for p in parent_path.split("."):
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        else:
            parent = model
            attr = parts[0]
        setattr(parent, attr, LoRALinear(module, r, alpha))
        count += 1

    for n, param in model.named_parameters():
        if "lora_" not in n:
            param.requires_grad = False
    return count


def lora_param_count(model: nn.Module) -> Dict[str, int]:
    """Return counts of trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
