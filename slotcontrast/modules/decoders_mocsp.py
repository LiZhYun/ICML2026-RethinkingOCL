"""MOCSP (D3) decoder extensions.

Subclasses of ``MLPDecoder`` that additionally expose per-slot
pre-mix reconstructions for the D3 MOCSP loss.
"""
from typing import Dict, Optional

import timm.layers.pos_embed
import torch
from torch import nn

from slotcontrast.modules.decoders import MLPDecoder


class MOCSPMLPDecoder(MLPDecoder):
    """MLPDecoder variant that also exposes per-slot pre-mix reconstructions.

    Output dict keys:
      - ``reconstruction`` : ``[B, (T,) P, outp_dim]``  — UNCHANGED
      - ``masks``          : ``[B, (T,) K, P]``          — UNCHANGED
      - ``recons_per_slot``: ``[B, (T,) K, P, outp_dim]`` — NEW, consumed by
        ``MOCSPLoss`` for per-slot held-out regression against a frozen
        DINO target.

    The parent's ``reconstruction`` / ``masks`` outputs remain byte-identical
    to ``MLPDecoder`` (same parameters, same computation); this subclass only
    adds a new output key carrying the un-mixed per-slot features.
    """

    def forward(
        self,
        slots: torch.Tensor,
        existence_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Video path: delegate to per-frame forward + stack.
        if slots.ndim == 4:
            B, T, n_slots, D = slots.shape
            outs = [
                self.forward(
                    slots[:, t],
                    existence_mask[:, t] if existence_mask is not None else None,
                )
                for t in range(T)
            ]
            return {
                "reconstruction": torch.stack([o["reconstruction"] for o in outs], dim=1),
                "masks": torch.stack([o["masks"] for o in outs], dim=1),
                "recons_per_slot": torch.stack(
                    [o["recons_per_slot"] for o in outs], dim=1
                ),
            }

        bs, n_slots, dims = slots.shape

        if not self.training and self.eval_output_size is not None:
            pos_emb = timm.layers.pos_embed.resample_abs_pos_embed(
                self.pos_emb.squeeze(1),
                new_size=self.eval_output_size,
                num_prefix_tokens=0,
            ).unsqueeze(1)
        else:
            pos_emb = self.pos_emb

        slots_e = slots.view(bs, n_slots, 1, dims).expand(
            bs, n_slots, pos_emb.shape[2], dims
        )
        slots_e = slots_e + pos_emb
        recons, alpha = self.mlp(slots_e).split((self.outp_dim, 1), dim=-1)
        # recons: [B, K, P, outp_dim]; alpha: [B, K, P, 1]

        if existence_mask is not None:
            mask = existence_mask.view(bs, n_slots, 1, 1)
            alpha = alpha.masked_fill(mask == 0, float("-inf"))

        masks = torch.softmax(alpha, dim=1)  # [B, K, P, 1]
        recon = torch.sum(recons * masks, dim=1)  # [B, P, outp_dim]

        return {
            "reconstruction": recon,
            "masks": masks.squeeze(-1),  # [B, K, P]
            "recons_per_slot": recons,    # [B, K, P, outp_dim]
        }


class KQueryMLPDecoder(MOCSPMLPDecoder):
    """Ablation decoder: replace incoming slots with K learnable queries.

    Used as a control for MOCSP: if slot conditioning is the mechanism
    driving MOCSP's signal, this ablation should perform substantially
    worse than ``MOCSPMLPDecoder`` because it discards all slot-attention
    output and uses fixed per-slot queries that carry no video-conditional
    information.

    The mask/alpha head still sees distinct queries per slot index, so it
    can still learn a meaningful per-slot spatial prior — but without any
    input-conditioned slot content.

    Args:
        inp_dim / outp_dim / hidden_dims / n_patches: same as ``MLPDecoder``.
        n_slots: number of learnable queries (must match the system's slot count).
    """

    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims,
        n_patches: int,
        n_slots: int,
        activation: str = "relu",
        eval_output_size=None,
        frozen: bool = False,
    ):
        super().__init__(
            inp_dim=inp_dim,
            outp_dim=outp_dim,
            hidden_dims=hidden_dims,
            n_patches=n_patches,
            activation=activation,
            eval_output_size=eval_output_size,
            frozen=frozen,
        )
        self.n_slots = int(n_slots)
        # Learnable per-slot queries that replace slot-attention output.
        self.k_queries = nn.Parameter(
            torch.randn(1, self.n_slots, inp_dim) * (inp_dim ** -0.5)
        )

    def forward(
        self,
        slots: torch.Tensor,
        existence_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # slots is ignored — we substitute our learned queries. The shape
        # of slots still determines B, T, K for broadcasting.
        if slots.ndim == 4:
            B, T, K, _ = slots.shape
            if K != self.n_slots:
                raise ValueError(
                    f"KQueryMLPDecoder.n_slots={self.n_slots} does not match "
                    f"incoming K={K}."
                )
            q = self.k_queries.unsqueeze(0).expand(B, T, K, self.k_queries.shape[-1])
            return MOCSPMLPDecoder.forward(self, q, existence_mask)

        B, K, _ = slots.shape
        if K != self.n_slots:
            raise ValueError(
                f"KQueryMLPDecoder.n_slots={self.n_slots} does not match "
                f"incoming K={K}."
            )
        q = self.k_queries.expand(B, K, self.k_queries.shape[-1])
        return MOCSPMLPDecoder.forward(self, q, existence_mask)
