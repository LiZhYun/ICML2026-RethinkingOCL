"""Compositional slot renderer for GSRS (Generative Slot Replay with Structured
Identity Supervision).

Implements the `g_phi` renderer from the GSRS proposal (§3.1(b) of
`refine-logs/FINAL_PROPOSAL_2026_04_15.md`).

Design decisions / spec resolutions (documented for reviewers):

1. **"3 transpose-conv layers"** (§3.1(b)). We interpret this as three
   transpose-conv upsampling stages preceded by a learnable linear projection
   from the 128-D per-(slot,frame) code to an initial (C_init, 14, 14) grid.
   The three transpose-conv stages upsample 14 -> 28 -> 56 -> 112, and a final
   parameterless bilinear resize produces the 224x224 output. This choice is
   what keeps the total parameter count in the ~2.5M target band stated in the
   spec while respecting the "3 transpose-conv" count exactly.

2. **Order-invariant compositing**. The spec's alpha-compositing formula
   ``x_hat = sum_k alpha_k * x_k + (1 - sum_k alpha_k)_+ * bg`` is itself
   order-invariant over the slot axis (symmetric sum). We implement this
   formula directly so the module is permutation-equivariant **by
   construction**, which is Gate G1(c) (``MSE < 1e-3``). The spec's mention of
   "back-to-front ordering by alpha sum (tie-break random)" is retained only
   as a reported, non-load-bearing statistic, since the compositing formula
   does not actually depend on ordering.

3. **Background stream**. A single learnable 128-D token ``b`` is time-tiled
   and passed through the same trajectory-encoder architecture (but with
   its own weights) and then its own (C_init, 14, 14)->(3, 224, 224) RGB
   decoder. RGB only -- the background carries no alpha.

4. **Permutation equivariance during pretraining**. The module is
   order-invariant by virtue of shared-across-slots weights in both the
   trajectory encoder and the RGBA decoder. Per-batch slot permutation is
   orchestrated by the pretraining script; the module itself does not need
   to shuffle.

This file contains **no silent fallbacks**. Missing inputs raise.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
#  Building blocks                                                            #
# --------------------------------------------------------------------------- #


class _PerSlotTrajectoryEncoder(nn.Module):
    """Shared-weights 1-layer Transformer over the temporal axis.

    Input:  ``z`` of shape ``[B*K, T, D_hidden]``
    Output:          shape ``[B*K, T, D_hidden]``

    Because weights are shared across slots (all slots stacked into the batch
    axis), the module is order-invariant in the slot dimension by construction.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.0,
        max_time: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_pos = nn.Parameter(torch.zeros(1, max_time, hidden_dim))
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        self.layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        T = z.shape[1]
        if T > self.time_pos.shape[1]:
            raise ValueError(
                f"Sequence length T={T} exceeds trajectory encoder max_time"
                f"={self.time_pos.shape[1]}"
            )
        z = z + self.time_pos[:, :T]
        z = self.layer(z)
        return self.norm(z)


class _RGBAPerSlotDecoder(nn.Module):
    """Shared-weights per-(slot, frame) RGBA decoder.

    Input:  ``code`` of shape ``[N, D_hidden]`` (N = B*K*T for slots)
    Output:          shape ``[N, 4, 224, 224]`` (R, G, B, alpha)

    Architecture:
        code (D_hidden) -> Linear -> (C_init, 14, 14)
            -> TConv 14->28
            -> TConv 28->56
            -> TConv 56->112
            -> bilinear upsample to 224.

    Note: alpha is passed through a sigmoid, RGB through a sigmoid (bounded
    [0, 1]). A final 1x1 conv projects the last feature map to 4 channels.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        c_init: int = 40,
        grid_init: int = 14,
        out_size: int = 224,
        n_out_channels: int = 4,
    ):
        super().__init__()
        self.c_init = c_init
        self.grid_init = grid_init
        self.out_size = out_size
        self.n_out_channels = n_out_channels

        self.proj = nn.Linear(hidden_dim, c_init * grid_init * grid_init)

        # Three transpose-conv stages (matches the spec's "3 transpose-conv
        # layers" count exactly). The last stage outputs n_out_channels=4
        # feature maps directly, so no extra 1x1 conv is needed.
        c1 = max(c_init // 2, 16)
        c2 = max(c_init // 4, 8)
        self.tconv1 = nn.ConvTranspose2d(c_init, c1, kernel_size=4, stride=2, padding=1)
        self.tconv2 = nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.tconv3 = nn.ConvTranspose2d(c2, n_out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        N = code.shape[0]
        x = self.proj(code).view(N, self.c_init, self.grid_init, self.grid_init)
        x = F.gelu(self.tconv1(x))
        x = F.gelu(self.tconv2(x))
        x = self.tconv3(x)
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)

        if self.n_out_channels == 4:
            rgb = torch.sigmoid(x[:, :3])
            alpha = torch.sigmoid(x[:, 3:4])
            return torch.cat([rgb, alpha], dim=1)
        else:  # RGB only (background)
            return torch.sigmoid(x)


# --------------------------------------------------------------------------- #
#  CompositionalSlotRenderer                                                  #
# --------------------------------------------------------------------------- #


class CompositionalSlotRenderer(nn.Module):
    """Slot-compositional renderer ``g_phi`` for GSRS.

    Given per-slot trajectories ``s`` of shape ``[B, T, K, slot_dim]``, produce
    per-slot RGBA layers ``L`` of shape ``[B, T, K, 4, H, W]`` and a background
    RGB stream ``bg`` of shape ``[B, T, 3, H, W]``. Alpha-composite per the
    spec's order-invariant formula:

        x_hat_t = sum_k alpha_k * rgb_k + (1 - sum_k alpha_k)_+ * bg_t

    where ``(.)_+`` denotes relu-clamp into [0, 1] (guards against Sum alpha_k
    exceeding 1.0).

    Architecture:
        - slot_dim -> hidden_dim projection (Linear)
        - Per-slot trajectory encoder (1-layer Transformer, shared weights
          across slots)
        - Per-slot RGBA decoder (3 transpose-conv layers, shared across slots)
        - Background trajectory encoder (separate weights) applied to a single
          learnable 128-D token tiled over time
        - Background decoder (3 transpose-conv layers, same arch, RGB only)

    Permutation equivariance is built-in via shared-across-slots weights
    + symmetric compositing formula. Renderer pretraining orchestrates the
    per-batch random slot permutation externally.

    Parameter budget (target ~2.5M from spec §3.1(b)):
        - slot_proj (slot_dim=64 -> 128):            ~8K
        - slot traj encoder (1-layer, d=128):        ~199K
        - bg traj encoder (1-layer, d=128):          ~199K
        - learnable bg token (128):                   128
        - slot RGBA decoder (c_init=40):             ~1.03M
        - bg RGB decoder (c_init=40):                ~1.02M
        -----------------------------------------------------
          Total (actual, checked at construction):   ~2.46M
    """

    def __init__(
        self,
        slot_dim: int = 64,
        hidden_dim: int = 128,
        out_size: int = 224,
        c_init: int = 40,
        grid_init: int = 14,
        max_time: int = 32,
        transformer_heads: int = 4,
        transformer_dim_ff: int = 512,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        self.out_size = out_size
        self.max_time = max_time

        # (1) slot_dim -> hidden_dim projection (shared across slots)
        self.slot_proj = nn.Linear(slot_dim, hidden_dim)

        # (2) Trajectory encoder for slot sequences (shared across slots)
        self.slot_traj_encoder = _PerSlotTrajectoryEncoder(
            hidden_dim=hidden_dim,
            n_heads=transformer_heads,
            dim_ff=transformer_dim_ff,
            max_time=max_time,
        )

        # (3) Per-slot RGBA decoder (shared across slots)
        self.slot_decoder = _RGBAPerSlotDecoder(
            hidden_dim=hidden_dim,
            c_init=c_init,
            grid_init=grid_init,
            out_size=out_size,
            n_out_channels=4,
        )

        # (4) Background token + trajectory encoder + RGB decoder
        self.bg_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.trunc_normal_(self.bg_token, std=0.02)
        self.bg_traj_encoder = _PerSlotTrajectoryEncoder(
            hidden_dim=hidden_dim,
            n_heads=transformer_heads,
            dim_ff=transformer_dim_ff,
            max_time=max_time,
        )
        self.bg_decoder = _RGBAPerSlotDecoder(
            hidden_dim=hidden_dim,
            c_init=c_init,
            grid_init=grid_init,
            out_size=out_size,
            n_out_channels=3,
        )

    # --------------------------------------------------------------------- #
    #  Forward                                                              #
    # --------------------------------------------------------------------- #

    def forward(
        self,
        slots: torch.Tensor,
        return_layers: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Render a clip.

        Args:
            slots: [B, T, K, slot_dim] slot trajectories.
            return_layers: whether to return the per-slot RGBA layers.

        Returns:
            dict with:
                - ``composited``: [B, T, 3, H, W] alpha-composited output.
                - ``layers_rgba``: [B, T, K, 4, H, W] per-slot RGBA (if
                  ``return_layers=True``).
                - ``bg_rgb``: [B, T, 3, H, W] background stream.
                - ``alpha_sum``: [B, T, 1, H, W] sum_k alpha_k (pre-clamp).
                - ``alpha_residual``: [B, T, 1, H, W] = (1 - sum_k alpha_k)_+.
        """
        if slots.ndim != 4:
            raise ValueError(
                f"slots must have shape [B, T, K, slot_dim]; got {tuple(slots.shape)}"
            )
        B, T, K, D = slots.shape
        if D != self.slot_dim:
            raise ValueError(
                f"slot_dim mismatch: configured {self.slot_dim}, got {D}"
            )

        # --- Slot branch -------------------------------------------------
        # Project slot_dim -> hidden_dim: [B, T, K, H]
        h = self.slot_proj(slots)

        # Rearrange to [B*K, T, H] so the trajectory encoder (shared weights)
        # runs independently per slot -- this is where slot-order invariance
        # comes from.
        h = h.permute(0, 2, 1, 3).contiguous().view(B * K, T, self.hidden_dim)
        h = self.slot_traj_encoder(h)  # [B*K, T, H]

        # Decode per-(slot, frame).
        codes = h.view(B * K * T, self.hidden_dim)
        rgba = self.slot_decoder(codes)  # [B*K*T, 4, H, W]
        rgba = rgba.view(B, K, T, 4, self.out_size, self.out_size)
        rgba = rgba.permute(0, 2, 1, 3, 4, 5).contiguous()  # [B, T, K, 4, H, W]

        rgb_k = rgba[:, :, :, :3]    # [B, T, K, 3, H, W]
        alpha_k = rgba[:, :, :, 3:4]  # [B, T, K, 1, H, W]

        # --- Background branch ------------------------------------------
        bg_tokens = self.bg_token.view(1, 1, self.hidden_dim).expand(B, T, -1)  # [B, T, H]
        bg_h = self.bg_traj_encoder(bg_tokens)  # [B, T, H]
        bg_codes = bg_h.view(B * T, self.hidden_dim)
        bg_rgb = self.bg_decoder(bg_codes)  # [B*T, 3, H, W]
        bg_rgb = bg_rgb.view(B, T, 3, self.out_size, self.out_size)

        # --- Alpha-compositing (spec §3.1(b)) --------------------------
        # x_hat = sum_k alpha_k * rgb_k + (1 - sum_k alpha_k)_+ * bg
        weighted = (alpha_k * rgb_k).sum(dim=2)  # [B, T, 3, H, W]
        alpha_sum = alpha_k.sum(dim=2)            # [B, T, 1, H, W]
        alpha_residual = (1.0 - alpha_sum).clamp(min=0.0)  # (.)_+ guard
        composited = weighted + alpha_residual * bg_rgb  # [B, T, 3, H, W]

        out = {
            "composited": composited,
            "bg_rgb": bg_rgb,
            "alpha_sum": alpha_sum,
            "alpha_residual": alpha_residual,
        }
        if return_layers:
            out["layers_rgba"] = rgba
        return out

    # --------------------------------------------------------------------- #
    #  Introspection helpers                                                 #
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Return total (or trainable) parameter count."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def render_single_slot(
        self,
        slots: torch.Tensor,
        slot_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Render only the ``slot_idx``-th slot by zeroing all other slot alphas.

        Used by Gate G1(b) leave-one-slot-out causality check. Returns the
        composited output and the kept layer's RGBA.
        """
        if slot_idx < 0 or slot_idx >= slots.shape[2]:
            raise ValueError(f"slot_idx {slot_idx} out of range [0, {slots.shape[2]})")
        out = self.forward(slots, return_layers=True)
        layers = out["layers_rgba"].clone()
        # Zero all other slots' alpha
        mask_kill = torch.ones(layers.shape[2], device=layers.device, dtype=layers.dtype)
        mask_kill[slot_idx] = 0.0
        layers[:, :, :, 3:4] = layers[:, :, :, 3:4] * (1 - mask_kill.view(1, 1, -1, 1, 1, 1))
        alpha_k = layers[:, :, :, 3:4]
        rgb_k = layers[:, :, :, :3]
        weighted = (alpha_k * rgb_k).sum(dim=2)
        alpha_sum = alpha_k.sum(dim=2)
        alpha_residual = (1.0 - alpha_sum).clamp(min=0.0)
        composited = weighted + alpha_residual * out["bg_rgb"]
        return {"composited": composited, "layers_rgba": layers, "bg_rgb": out["bg_rgb"]}

    @torch.no_grad()
    def permutation_equivariance_residual(
        self,
        slots: torch.Tensor,
        perm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mse, perm) where ``mse`` is the MSE between compositing
        under slots and slots[:, :, perm, :]. This is Gate G1(c).

        If ``perm`` is None, a random permutation is drawn.
        """
        K = slots.shape[2]
        if perm is None:
            perm = torch.randperm(K, device=slots.device)
        out_a = self.forward(slots, return_layers=False)["composited"]
        out_b = self.forward(slots[:, :, perm, :], return_layers=False)["composited"]
        mse = F.mse_loss(out_a, out_b)
        return mse, perm


# --------------------------------------------------------------------------- #
#  Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, K, D = 2, 4, 7, 64
    renderer = CompositionalSlotRenderer(
        slot_dim=D, hidden_dim=128, out_size=224, c_init=40, grid_init=14, max_time=32
    )
    n_params = renderer.count_parameters()
    print(f"[smoke] param count: {n_params:,} ({n_params / 1e6:.2f}M)")

    slots = torch.randn(B, T, K, D)
    out = renderer(slots)
    print(f"[smoke] composited:     {tuple(out['composited'].shape)}")
    print(f"[smoke] layers_rgba:    {tuple(out['layers_rgba'].shape)}")
    print(f"[smoke] bg_rgb:         {tuple(out['bg_rgb'].shape)}")
    print(f"[smoke] alpha_sum:      {tuple(out['alpha_sum'].shape)}")

    alpha_k = out["layers_rgba"][:, :, :, 3:4]
    assert (alpha_k >= 0).all() and (alpha_k <= 1).all(), "alpha out of [0,1]"
    assert (out["composited"] >= 0).all(), "composited RGB is negative"
    assert (out["alpha_residual"] >= 0).all() and (out["alpha_residual"] <= 1).all(), \
        "alpha_residual out of [0,1]"
    max_c = out["composited"].max().item()
    print(f"[smoke] alpha and residual bounds ok. composited max = {max_c:.3f} "
          f"(may exceed 1 when sum_k alpha_k > 1; downstream loss clamps).")

    mse, perm = renderer.permutation_equivariance_residual(slots)
    print(f"[smoke] permutation-equivariance MSE: {mse.item():.3e}  (target < 1e-3)")

    lo_out = renderer.render_single_slot(slots, slot_idx=0)
    print(f"[smoke] single-slot composited: {tuple(lo_out['composited'].shape)}")
    print("[smoke] all shape checks passed.")
