"""Cutie-style object-query memory bank for slot identity re-retrieval (H1).

Inspired by Cheng et al., "Putting the Object Back into Video Object Segmentation"
(Cutie, CVPR 2024, arXiv:2310.12982), §3 "Object-Level Memory Reading". In Cutie
the object memory is a small set of per-object vectors that cross-attend to the
pixel feature memory — proven to beat XMem's pixel-only memory on distractor-heavy
sequences. Here we port the primitive to slot-centric video OCL: memory entries
are directly per-slot feature vectors (each stored with a timestamp and the
attention mass the slot carried at the frame it was written).

This module has ZERO learned parameters. It is a pure inference-time structure
that the ``CutieSlotPredictor`` consults to rescue identity when the direct
slot-to-slot Hungarian match is weak (post-occlusion re-identification).

No silent fallback policy
-------------------------
When the memory bank is empty (at clip start), ``read`` returns memory cost
``+inf`` for every query slot. The ``CutieSlotPredictor`` combines memory cost
with the direct Hungarian cost via elementwise min, so ``+inf`` memory cost
cleanly reduces to standard Hungarian — no hidden defaults or shims.

Eviction policy (documented per spec)
-------------------------------------
FIFO with oldest-timestamp-first: if two entries share the same timestamp,
break ties by the lower attention-mass they held when written (remove the
less confident one). This is pure inference structure; no gradients flow.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ObjectQueryMemoryBank(nn.Module):
    """FIFO per-slot feature memory with cosine-similarity retrieval.

    State
    -----
    Per batch element ``b`` we maintain three aligned lists (ragged across the
    batch dimension, bounded above by ``capacity`` per element):
      - ``_features[b]``  : List[Tensor[D]]   — slot feature snapshots
      - ``_timestamps[b]``: List[int]          — frame index when written
      - ``_attn_mass[b]`` : List[float]        — attention mass at write time

    Writes are keyed by slot index: at frame ``t`` the bank receives
    ``slots_t[B, K, D]`` and an ``attention_mass_t[B, K]`` vector (per-slot
    fraction of patch attention the slot captured on that frame). We insert one
    entry per ``(b, k)``; if the batch's buffer exceeds ``capacity``, evict the
    single oldest-timestamp entry (ties broken by lowest ``attn_mass``).

    Reads take query slots ``query[B, K, D]`` and, for each query slot, return
    the top-``top_k`` memory entries by cosine similarity (one per-batch
    per-query triple). Empty slots in the bank pad with +inf distance (= -inf
    similarity) so downstream elementwise-min with the direct Hungarian cost
    recovers the no-memory baseline without a branch.

    Args:
        capacity: Max memory entries per batch element. Default 64 per spec.
        top_k: Number of nearest memory entries returned per query. Default 3.
        similarity: Similarity metric — currently only ``'cosine'`` is
            implemented. Explicit argument so future extensions are additive.
    """

    def __init__(
        self,
        capacity: int = 64,
        top_k: int = 3,
        similarity: str = "cosine",
    ):
        super().__init__()
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if similarity != "cosine":
            raise NotImplementedError(
                f"similarity='{similarity}' not implemented; only 'cosine' is "
                "supported (hard-raise rather than silent fallback)."
            )
        self.capacity = int(capacity)
        self.top_k = int(top_k)
        self.similarity = similarity

        # Per-batch ragged state. Populated by ``reset`` on each new clip.
        self._features: List[List[torch.Tensor]] = []
        self._timestamps: List[List[int]] = []
        self._attn_mass: List[List[float]] = []
        self._batch_size: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, batch_size: int) -> None:
        """Clear state for a new video clip with the given batch size."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._batch_size = int(batch_size)
        self._features = [[] for _ in range(self._batch_size)]
        self._timestamps = [[] for _ in range(self._batch_size)]
        self._attn_mass = [[] for _ in range(self._batch_size)]

    def occupancy(self) -> List[int]:
        """Per-batch number of memory entries currently held."""
        return [len(buf) for buf in self._features]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    @torch.no_grad()
    def write(
        self,
        slots_t: torch.Tensor,
        attention_mass_t: Optional[torch.Tensor],
        t: int,
    ) -> None:
        """Write per-slot snapshots at frame ``t`` into the bank.

        Args:
            slots_t: ``[B, K, D]`` slot features for the current frame.
            attention_mass_t: ``[B, K]`` attention mass (or confidence) per
                slot. Used as the eviction tie-breaker and stored alongside the
                feature. If ``None``, a uniform value of ``1.0`` is used — this
                still permits FIFO but removes the mass-aware tie-break.
            t: integer frame index.
        """
        if slots_t.ndim != 3:
            raise ValueError(
                f"slots_t must be [B, K, D], got shape {tuple(slots_t.shape)}"
            )
        B, K, _ = slots_t.shape
        if B != self._batch_size:
            raise RuntimeError(
                f"Bank was reset for batch_size={self._batch_size} but write "
                f"received batch={B}. Call reset(B) before the first write of "
                "each new clip."
            )
        if attention_mass_t is not None:
            if attention_mass_t.shape[:2] != slots_t.shape[:2]:
                raise ValueError(
                    "attention_mass_t must have shape [B, K] matching slots_t; "
                    f"got {tuple(attention_mass_t.shape)} vs "
                    f"{tuple(slots_t.shape[:2])}"
                )

        feats = slots_t.detach()
        for b in range(B):
            for k in range(K):
                am = (
                    float(attention_mass_t[b, k].item())
                    if attention_mass_t is not None
                    else 1.0
                )
                self._features[b].append(feats[b, k].clone())
                self._timestamps[b].append(int(t))
                self._attn_mass[b].append(am)
            self._evict_to_capacity(b)

    def _evict_to_capacity(self, b: int) -> None:
        """Remove oldest (then lowest-attn) entries until size <= capacity."""
        buf_f = self._features[b]
        buf_t = self._timestamps[b]
        buf_a = self._attn_mass[b]
        while len(buf_f) > self.capacity:
            # Find oldest timestamp; break ties by lowest attention mass.
            oldest_ts = min(buf_t)
            # Collect indices at the oldest timestamp.
            cand = [i for i, ts in enumerate(buf_t) if ts == oldest_ts]
            # Pick the lowest-attn-mass among them.
            victim = min(cand, key=lambda i: buf_a[i])
            buf_f.pop(victim)
            buf_t.pop(victim)
            buf_a.pop(victim)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    @torch.no_grad()
    def read(
        self,
        query_slots_t: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve top-k memory entries per query slot by cosine similarity.

        Args:
            query_slots_t: ``[B, K, D]`` query slots.
            top_k: Optional override for ``self.top_k``.

        Returns:
            Tuple of
              - ``memory_feats``: ``[B, K, top_k, D]`` retrieved entries. For
                padded (empty) slots, entries are zero tensors and the
                corresponding ``valid`` flag is False.
              - ``memory_ts``: ``[B, K, top_k]`` long tensor of timestamps;
                -1 where the entry is a pad (no memory available).
              - ``valid``: ``[B, K, top_k]`` boolean mask; True = real memory
                entry, False = pad. Downstream cost computations treat False
                positions as ``+inf`` cost so they never win a min-reduction.
        """
        if query_slots_t.ndim != 3:
            raise ValueError(
                f"query_slots_t must be [B, K, D], got {tuple(query_slots_t.shape)}"
            )
        B, K, D = query_slots_t.shape
        if B != self._batch_size:
            raise RuntimeError(
                f"Bank was reset for batch_size={self._batch_size} but read "
                f"received batch={B}."
            )

        topk = int(top_k) if top_k is not None else self.top_k
        device = query_slots_t.device
        dtype = query_slots_t.dtype

        out_feats = torch.zeros(B, K, topk, D, device=device, dtype=dtype)
        out_ts = torch.full((B, K, topk), -1, device=device, dtype=torch.long)
        valid = torch.zeros(B, K, topk, device=device, dtype=torch.bool)

        # Normalized query for cosine similarity.
        q_norm = F.normalize(query_slots_t, dim=-1)

        for b in range(B):
            n_mem = len(self._features[b])
            if n_mem == 0:
                continue
            # Stack memory entries for this batch element once.
            mem = torch.stack(self._features[b], dim=0).to(device=device, dtype=dtype)  # [N_mem, D]
            mem_norm = F.normalize(mem, dim=-1)
            # [K, D] x [D, N_mem] -> [K, N_mem]
            sims = q_norm[b] @ mem_norm.transpose(0, 1)
            effective_k = min(topk, n_mem)
            top_sim, top_idx = sims.topk(effective_k, dim=-1)  # [K, effective_k]
            # Gather feats + timestamps.
            sel_feats = mem[top_idx]  # [K, effective_k, D]
            ts_tensor = torch.tensor(
                self._timestamps[b], device=device, dtype=torch.long
            )
            sel_ts = ts_tensor[top_idx]  # [K, effective_k]
            out_feats[b, :, :effective_k] = sel_feats
            out_ts[b, :, :effective_k] = sel_ts
            valid[b, :, :effective_k] = True

        return out_feats, out_ts, valid


__all__ = ["ObjectQueryMemoryBank"]
