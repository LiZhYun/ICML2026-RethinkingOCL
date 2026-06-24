"""Replay machinery for GSRS (Generative Slot Replay with Structured Identity
Supervision).

Implements (spec refs to `refine-logs/FINAL_PROPOSAL_2026_04_15.md`):

- ``DonorBank``       (§3.2 birth family; FIFO of donor (slot_trajectory,
  appearance_crop) pairs)
- ``DistinguishabilityFilter`` (§3.3)
- ``InterventionSampler``      (§3.2 five intervention families)
- ``LeakageClassifier``        (§3.4 automatic audit #1)
- ``CrossFeatureRetrievalAgreement`` (§3.4 automatic audit #2)
- ``LocalityCheck``            (§3.4 automatic audit #3)

**No silent fallbacks.** Missing DINOv2/SigLIP or donor-bank underflow raise
clean errors rather than degrading to a no-op. The SigLIP encoder is optional
per spec and falls back to a *second DINOv2 view* as the spec explicitly
permits ("or a second DINOv2 view as fallback if SigLIP not available").

The module imports the renderer lazily to avoid circular imports during
pretraining, where the renderer is the object being trained.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
#  Shared constants                                                           #
# --------------------------------------------------------------------------- #


# Shared contract between Part A (intervention sampler) and Part B
# (``GSRSIdentityLoss`` + ``OpenSetIdentityHead``). This ordering MUST match
# ``networks.OpenSetIdentityHead.CATEGORIES`` — the loss uses the index to
# look up the CE target from ``open_set_probs``. Exposed at module scope so
# downstream code (data collator, loss) can ``from slotcontrast.modules.replay
# import EVENT_TO_INDEX`` and avoid redefining the mapping.
EVENT_TO_INDEX: Dict[str, int] = {
    "source": 0,
    "dormant": 1,
    "null": 2,
    "born": 3,
}


# --------------------------------------------------------------------------- #
#  Data structures                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class DonorEntry:
    """One donor banked for the `birth` intervention family."""

    slot_trajectory: torch.Tensor  # [T, slot_dim]
    appearance_crop: torch.Tensor  # [3, H_crop, W_crop] (nominal; not used if
                                   # intervention uses full trajectory render)
    origin_clip_id: str = ""


@dataclass
class InterventionRecord:
    """Metadata describing one applied intervention."""

    family: str  # one of {'same', 'swap', 'gap-return', 'kill', 'birth'}
    slot_indices: Tuple[int, ...]  # which slots in the source clip were manipulated
    swap_partner: Optional[int] = None     # for 'swap'
    gap_window: Optional[Tuple[int, int]] = None  # for 'gap-return', inclusive
    kill_time: Optional[int] = None         # for 'kill'
    birth_time: Optional[int] = None        # for 'birth'
    donor_id: Optional[str] = None          # for 'birth'
    extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  DonorBank                                                                  #
# --------------------------------------------------------------------------- #


class DonorBank:
    """FIFO queue of donor trajectories used for the `birth` intervention
    family (§3.2 item 5).

    Each entry is a ``DonorEntry``. Pushed clips' slot trajectories are stored
    with CPU-pinned tensors (to avoid GPU retention). On sampling, the caller
    is responsible for moving the tensor to the correct device.

    No silent fallback: sampling when empty raises.
    """

    def __init__(self, capacity: int = 500):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self._q: Deque[DonorEntry] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._q)

    def push(self, entry: DonorEntry) -> None:
        if entry.slot_trajectory.ndim != 2:
            raise ValueError(
                f"slot_trajectory must be [T, slot_dim], got "
                f"{tuple(entry.slot_trajectory.shape)}"
            )
        # detach + cpu to avoid GPU retention across bank cycles
        entry = DonorEntry(
            slot_trajectory=entry.slot_trajectory.detach().to("cpu"),
            appearance_crop=entry.appearance_crop.detach().to("cpu"),
            origin_clip_id=entry.origin_clip_id,
        )
        self._q.append(entry)

    def sample(self, rng: Optional[random.Random] = None) -> DonorEntry:
        if len(self._q) == 0:
            raise RuntimeError(
                "DonorBank is empty; cannot sample a donor. "
                "The GSRS pipeline contract requires the bank to be warmed "
                "before the first `birth` intervention is requested."
            )
        rng = rng if rng is not None else random
        return rng.choice(self._q)


# --------------------------------------------------------------------------- #
#  DistinguishabilityFilter                                                   #
# --------------------------------------------------------------------------- #


class DistinguishabilityFilter:
    """Implements §3.3. A pair (A, B) is distinguishable iff:

    - ``cos(s_bar_A, s_bar_B) < 0.6`` (time-averaged slot-vector cosine sim).
    - Centroid separation > 8 px **at all common-visible frames**.
    - Mean visibility > 0.5 (averaged across clip) for both slots.

    Visibility is derived from per-slot alpha masks: a slot is "visible" at
    frame t if its alpha mass at t exceeds ``visibility_alpha_thresh``.

    Centroids are computed as alpha-weighted (y, x) pixel means.
    """

    def __init__(
        self,
        cos_thresh: float = 0.6,
        centroid_sep_px: float = 8.0,
        visibility_mean_thresh: float = 0.5,
        visibility_alpha_thresh: float = 0.05,
    ):
        self.cos_thresh = float(cos_thresh)
        self.centroid_sep_px = float(centroid_sep_px)
        self.visibility_mean_thresh = float(visibility_mean_thresh)
        self.visibility_alpha_thresh = float(visibility_alpha_thresh)

    # -------- utilities ----------------------------------------------------
    @staticmethod
    def _alpha_centroids(alpha: torch.Tensor) -> torch.Tensor:
        """alpha: [..., 1, H, W] -> centroids [..., 2] (y, x) in pixel coords."""
        if alpha.shape[-3] != 1:
            raise ValueError(f"alpha must have 1 channel, got shape {tuple(alpha.shape)}")
        H, W = alpha.shape[-2], alpha.shape[-1]
        ys = torch.arange(H, device=alpha.device, dtype=alpha.dtype).view(1, 1, -1, 1)
        xs = torch.arange(W, device=alpha.device, dtype=alpha.dtype).view(1, 1, 1, -1)
        a2 = alpha.view(-1, 1, H, W)
        mass = a2.sum(dim=(-1, -2)).clamp(min=1e-6)
        cy = (a2 * ys).sum(dim=(-1, -2)) / mass
        cx = (a2 * xs).sum(dim=(-1, -2)) / mass
        c = torch.stack([cy.squeeze(-1), cx.squeeze(-1)], dim=-1)
        return c.view(*alpha.shape[:-3], 2)

    @staticmethod
    def _alpha_mass(alpha: torch.Tensor) -> torch.Tensor:
        """Return per-frame alpha mass normalised by image area (in [0,1])."""
        H, W = alpha.shape[-2], alpha.shape[-1]
        return alpha.sum(dim=(-1, -2, -3)) / (H * W)

    # -------- main entry point --------------------------------------------
    def distinguishable_pairs(
        self,
        slots: torch.Tensor,      # [T, K, D]
        layers_rgba: torch.Tensor,  # [T, K, 4, H, W]
    ) -> List[Tuple[int, int]]:
        """Return all (i, j) with i < j that pass §3.3."""
        if slots.ndim != 3 or layers_rgba.ndim != 5:
            raise ValueError(
                "distinguishable_pairs expects slots [T, K, D] and "
                "layers_rgba [T, K, 4, H, W]"
            )
        T, K, D = slots.shape
        if T != layers_rgba.shape[0] or K != layers_rgba.shape[1]:
            raise ValueError("slots and layers_rgba must share T and K")

        alpha = layers_rgba[:, :, 3:4]  # [T, K, 1, H, W]

        # Time-averaged per-slot vector for cosine similarity (§3.3 s_bar).
        s_bar = slots.mean(dim=0)                  # [K, D]
        s_bar_n = F.normalize(s_bar, dim=-1)
        cos_sim = s_bar_n @ s_bar_n.t()             # [K, K]

        # Per-frame centroid and visibility mask.
        centroid = self._alpha_centroids(alpha)    # [T, K, 2]
        frame_mass = alpha.sum(dim=(-1, -2, -3)) / (alpha.shape[-1] * alpha.shape[-2])  # [T, K]
        visible = frame_mass > self.visibility_alpha_thresh  # [T, K]
        mean_vis = visible.float().mean(dim=0)     # [K]

        pairs: List[Tuple[int, int]] = []
        for i in range(K):
            for j in range(i + 1, K):
                if cos_sim[i, j].item() >= self.cos_thresh:
                    continue
                if mean_vis[i].item() <= self.visibility_mean_thresh:
                    continue
                if mean_vis[j].item() <= self.visibility_mean_thresh:
                    continue
                common = (visible[:, i] & visible[:, j])
                if not common.any():
                    # No common-visible frame -> the "at all common-visible
                    # frames" predicate is vacuously true; but a pair with no
                    # overlap cannot be meaningfully swapped, so we reject.
                    continue
                sep = (centroid[:, i] - centroid[:, j]).norm(dim=-1)
                min_sep = sep[common].min().item()
                if min_sep <= self.centroid_sep_px:
                    continue
                pairs.append((i, j))
        return pairs


# --------------------------------------------------------------------------- #
#  LeakageClassifier                                                          #
# --------------------------------------------------------------------------- #


class LeakageClassifier(nn.Module):
    """§3.4 automatic audit #1. 3-layer MLP over DINOv2(``x_hat``) patch-token
    statistics that tries to classify intervention family. Pass iff accuracy
    <= 1.1 * chance.

    The classifier is held **out** of the main training graph: it is trained
    on a small buffer of (dino_feat, family_label) pairs and queried on new
    samples. This module implements the MLP + the online trainer.
    """

    FAMILIES = ("same", "swap", "gap-return", "kill", "birth")

    def __init__(
        self,
        dino_feat_dim: int = 768,
        hidden_dim: int = 256,
        n_classes: int = 5,
        buffer_size: int = 2048,
        lr: float = 1e-3,
        train_every: int = 16,
        train_steps_per_call: int = 32,
    ):
        super().__init__()
        if n_classes != len(self.FAMILIES):
            raise ValueError(
                f"n_classes must equal {len(self.FAMILIES)} to match FAMILIES"
            )
        self.n_classes = n_classes
        self.mlp = nn.Sequential(
            nn.Linear(dino_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_classes),
        )
        self._opt = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self._buffer_feat: Deque[torch.Tensor] = deque(maxlen=buffer_size)
        self._buffer_label: Deque[int] = deque(maxlen=buffer_size)
        self.train_every = int(train_every)
        self.train_steps_per_call = int(train_steps_per_call)
        self._step = 0

    @staticmethod
    def family_index(family: str) -> int:
        try:
            return LeakageClassifier.FAMILIES.index(family)
        except ValueError as e:
            raise ValueError(f"Unknown intervention family: {family}") from e

    def chance(self) -> float:
        return 1.0 / self.n_classes

    def _pool(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: [B, N_patches, D] -> [B, D] via mean pool."""
        if feat.ndim != 3:
            raise ValueError(f"DINO feats must be [B, N, D]; got {tuple(feat.shape)}")
        return feat.mean(dim=1)

    def observe(self, dino_feat: torch.Tensor, family: str) -> None:
        """Push a (feat, label) pair into the classifier's online buffer.

        ``dino_feat``: [N_patches, D] or [1, N_patches, D].
        """
        if dino_feat.ndim == 2:
            dino_feat = dino_feat.unsqueeze(0)
        pooled = self._pool(dino_feat).detach().cpu()
        self._buffer_feat.append(pooled[0])
        self._buffer_label.append(self.family_index(family))
        self._step += 1
        if self._step % self.train_every == 0 and len(self._buffer_feat) >= 32:
            self._train_steps()

    def _train_steps(self) -> None:
        device = next(self.mlp.parameters()).device
        feats = torch.stack(list(self._buffer_feat), dim=0).to(device)
        labels = torch.tensor(list(self._buffer_label), device=device, dtype=torch.long)
        n = feats.shape[0]
        for _ in range(self.train_steps_per_call):
            idx = torch.randint(0, n, (min(64, n),), device=device)
            logits = self.mlp(feats[idx])
            loss = F.cross_entropy(logits, labels[idx])
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()

    @torch.no_grad()
    def predict_family_prob(self, dino_feat: torch.Tensor) -> torch.Tensor:
        pooled = self._pool(dino_feat)
        logits = self.mlp(pooled)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def held_out_accuracy(self) -> float:
        """Compute accuracy on the entire buffer (used as a cheap monitor).
        Rigorous held-out evaluation should be run externally."""
        if len(self._buffer_feat) < 2:
            return self.chance()
        device = next(self.mlp.parameters()).device
        feats = torch.stack(list(self._buffer_feat), dim=0).to(device)
        labels = torch.tensor(list(self._buffer_label), device=device, dtype=torch.long)
        logits = self.mlp(feats)
        pred = logits.argmax(dim=-1)
        return (pred == labels).float().mean().item()

    def passes_audit(self, tolerance: float = 1.1) -> bool:
        """§3.4 pass criterion: accuracy <= tolerance * chance."""
        return self.held_out_accuracy() <= tolerance * self.chance()


# --------------------------------------------------------------------------- #
#  CrossFeatureRetrievalAgreement                                             #
# --------------------------------------------------------------------------- #


class CrossFeatureRetrievalAgreement:
    """§3.4 automatic audit #2. Given two feature extractors ``f1`` and ``f2``
    (DINOv2 + SigLIP, or DINOv2 + second-view DINOv2 fallback per spec §3.4),
    retrieve the source object for each manipulated layer. Pass iff both
    extractors agree on the retrieved source.

    This class does NOT wrap the feature extractors themselves; the caller
    supplies two feature functions that take a cropped RGB tensor and return
    a [D]-vector embedding.
    """

    def __init__(
        self,
        f1: Callable[[torch.Tensor], torch.Tensor],
        f2: Callable[[torch.Tensor], torch.Tensor],
        top_k: int = 1,
    ):
        if f1 is None or f2 is None:
            raise ValueError(
                "CrossFeatureRetrievalAgreement requires two feature "
                "extractors. If SigLIP is unavailable, pass a second DINOv2 "
                "forward (e.g. a different augmentation view) per spec §3.4."
            )
        self.f1 = f1
        self.f2 = f2
        self.top_k = int(top_k)

    @torch.no_grad()
    def agreement_rate(
        self,
        manipulated_crops: Sequence[torch.Tensor],  # each [3, H, W]
        source_bank_crops: Sequence[torch.Tensor],  # each [3, H, W]
    ) -> Tuple[float, List[bool]]:
        """Return (rate, per_query flags) where rate = #agree / #queries.

        A manipulated crop's "retrieved source" is the argmax cosine
        neighbour in ``source_bank_crops``. Agreement = f1 and f2 pick the
        same source index.
        """
        if len(manipulated_crops) == 0:
            return 1.0, []
        if len(source_bank_crops) == 0:
            raise ValueError("source_bank_crops is empty; nothing to retrieve against")

        def _embed(fn: Callable, crops: Sequence[torch.Tensor]) -> torch.Tensor:
            embeds = [fn(c) for c in crops]
            stk = torch.stack(embeds, dim=0)
            return F.normalize(stk, dim=-1)

        q1 = _embed(self.f1, manipulated_crops)
        q2 = _embed(self.f2, manipulated_crops)
        b1 = _embed(self.f1, source_bank_crops)
        b2 = _embed(self.f2, source_bank_crops)

        sim1 = q1 @ b1.t()  # [Q, Bank]
        sim2 = q2 @ b2.t()
        ret1 = sim1.argmax(dim=-1)
        ret2 = sim2.argmax(dim=-1)

        flags = (ret1 == ret2).tolist()
        rate = float(sum(flags)) / max(1, len(flags))
        return rate, flags

    def passes_audit(
        self,
        manipulated_crops: Sequence[torch.Tensor],
        source_bank_crops: Sequence[torch.Tensor],
        pass_threshold: float = 1.0,
    ) -> bool:
        """Pass iff agreement rate >= pass_threshold (default: full agreement).

        Spec §3.4 item 2: "DINOv2 and SigLIP retrieve the same source object
        **for each manipulated object**." We use pass_threshold=1.0 by default
        to match the spec verbatim; callers can relax if desired.
        """
        rate, _ = self.agreement_rate(manipulated_crops, source_bank_crops)
        return rate >= pass_threshold


# --------------------------------------------------------------------------- #
#  LocalityCheck                                                              #
# --------------------------------------------------------------------------- #


class LocalityCheck:
    """§3.4 automatic audit #3. Non-manipulated objects' alpha masks should
    differ by <= 0.1 IoU between the source rendering and the manipulated
    rendering.
    """

    def __init__(self, iou_tolerance: float = 0.1, alpha_bin_thresh: float = 0.2):
        self.iou_tolerance = float(iou_tolerance)
        self.alpha_bin_thresh = float(alpha_bin_thresh)

    @staticmethod
    def _binarise(alpha: torch.Tensor, thresh: float) -> torch.Tensor:
        return (alpha > thresh).float()

    def iou(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a, b: binarised masks with identical shape."""
        inter = (a * b).sum(dim=(-1, -2, -3))
        union = ((a + b).clamp(max=1.0)).sum(dim=(-1, -2, -3))
        return inter / union.clamp(min=1e-6)

    def check(
        self,
        source_layers_rgba: torch.Tensor,  # [T, K, 4, H, W]
        manipulated_layers_rgba: torch.Tensor,
        manipulated_slot_indices: Sequence[int],
    ) -> Tuple[bool, torch.Tensor]:
        """Return (passes, iou_delta) where iou_delta is [K] per-slot IoU diff
        (NaN at manipulated slots to flag "not checked")."""
        if source_layers_rgba.shape != manipulated_layers_rgba.shape:
            raise ValueError("source and manipulated layers shapes must match")
        T, K = source_layers_rgba.shape[:2]
        src_alpha = self._binarise(source_layers_rgba[:, :, 3:4], self.alpha_bin_thresh)
        mod_alpha = self._binarise(manipulated_layers_rgba[:, :, 3:4], self.alpha_bin_thresh)

        ious: List[float] = []
        manip_set = set(manipulated_slot_indices)
        worst = 0.0
        for k in range(K):
            if k in manip_set:
                ious.append(float("nan"))
                continue
            iou_k = []
            for t in range(T):
                a = src_alpha[t, k]
                b = mod_alpha[t, k]
                iou_k.append(self.iou(a, b).item())
            mean_iou = sum(iou_k) / len(iou_k)
            ious.append(mean_iou)
            worst = max(worst, 1.0 - mean_iou)  # larger delta = worse
        passes = worst <= self.iou_tolerance
        return passes, torch.tensor(ious)


# --------------------------------------------------------------------------- #
#  InterventionSampler                                                        #
# --------------------------------------------------------------------------- #


class InterventionSampler:
    """Sample an intervention from ``I`` (§3.2) and apply it to a slot
    trajectory via the renderer.

    Returns a replay sample iff at least 2 of 3 automatic audits pass (§3.4).

    The sampler orchestrates:
        1. Apply the renderer ``g_phi`` to the source slots -> source layers.
        2. Sample a family from ``families`` with ``probs``.
        3. Compute the manipulated slot trajectory & layers.
        4. Apply corruption pipeline (JPEG/blur/color/dropout/resize).
        5. Run the 3 audits.
        6. Emit the replay sample if >= 2 audits pass.

    The corruption pipeline is a callable injected by the caller (it's a
    data-pipeline concern; keeping it here pluggable avoids this module
    depending on slotcontrast.data.transforms).
    """

    DEFAULT_FAMILIES: Tuple[str, ...] = ("same", "swap", "gap-return", "kill", "birth")
    DEFAULT_PROBS: Tuple[float, ...] = (0.35, 0.20, 0.25, 0.10, 0.10)

    def __init__(
        self,
        renderer: nn.Module,
        distinguishability_filter: DistinguishabilityFilter,
        donor_bank: Optional[DonorBank] = None,
        families: Sequence[str] = DEFAULT_FAMILIES,
        probs: Sequence[float] = DEFAULT_PROBS,
        corruption_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        leakage_classifier: Optional[LeakageClassifier] = None,
        retrieval_agreement: Optional[CrossFeatureRetrievalAgreement] = None,
        locality_check: Optional[LocalityCheck] = None,
        min_audits_passed: int = 2,
    ):
        if len(families) != len(probs):
            raise ValueError("families and probs must have equal length")
        ps = torch.tensor(list(probs), dtype=torch.float64)
        if not torch.isclose(ps.sum(), torch.tensor(1.0, dtype=torch.float64)):
            raise ValueError(f"probs must sum to 1.0; got {ps.sum().item()}")
        self.renderer = renderer
        self.dist_filter = distinguishability_filter
        self.donor_bank = donor_bank
        self.families = tuple(families)
        self.probs = tuple(float(p) for p in probs)
        self.corruption_fn = corruption_fn
        self.leakage_cls = leakage_classifier
        self.retrieval = retrieval_agreement
        self.locality = locality_check
        if min_audits_passed < 0 or min_audits_passed > 3:
            raise ValueError("min_audits_passed must be in [0, 3]")
        self.min_audits_passed = int(min_audits_passed)

    # -------- family sampling ---------------------------------------------
    def _sample_family(self, rng: random.Random) -> str:
        r = rng.random()
        cum = 0.0
        for fam, p in zip(self.families, self.probs):
            cum += p
            if r <= cum:
                return fam
        return self.families[-1]

    # -------- core API ----------------------------------------------------
    def emit(
        self,
        source_slots: torch.Tensor,  # [T, K, D]
        rng: Optional[random.Random] = None,
        dino_feat_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        source_bank_crops: Optional[Sequence[torch.Tensor]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Attempt to emit one replay sample.

        Returns ``None`` if audits fail (caller should skip).

        Args:
            source_slots: [T, K, D] trusted-teacher slot trajectories.
            rng: optional random.Random for reproducibility.
            dino_feat_fn: callable (rgb [3, H, W]) -> DINOv2 patch feats
                [N_patches, D]. Required for leakage audit.
            source_bank_crops: list of reference crops for retrieval audit.

        Returns:
            dict with:
                - ``x_hat``:            [T, 3, H, W] composited manipulated RGB
                - ``replay_labels_for_dustbin``: dict with ``'family'``,
                  ``'slot_indices'``, ``'source_K'``, ``'recovered_K_hint'``
                  (the main-training ``L_A`` loss constructs the rectangular
                   plan from this).
                - ``event_flags_for_open_set_head``: dict with per-slot
                  per-frame flags in {source, dormant, null, born}.
                - ``intervention_metadata``: ``InterventionRecord``.
                - ``audits``: dict of audit pass/fail booleans.
            or ``None`` if audits fail.
        """
        rng = rng if rng is not None else random
        if source_slots.ndim != 3:
            raise ValueError(f"source_slots must be [T, K, D]; got {tuple(source_slots.shape)}")
        T, K, D = source_slots.shape

        # 1) Render source (to supply both audit baseline and distinguishability
        #    measurements).
        with torch.no_grad():
            src_out = self.renderer(source_slots.unsqueeze(0))  # add batch dim
        src_layers = src_out["layers_rgba"][0]   # [T, K, 4, H, W]

        # 2) Sample family. If `swap` and no distinguishable pair -> fall back
        #    to `same` per §3.3.
        family = self._sample_family(rng)
        pairs = self.dist_filter.distinguishable_pairs(source_slots, src_layers)
        if family == "swap" and len(pairs) == 0:
            family = "same"

        # 3) Build manipulated slot trajectory.
        manip_slots, record = self._apply_family(source_slots, family, pairs, rng)

        # 4) Render manipulated.
        with torch.no_grad():
            mod_out = self.renderer(manip_slots.unsqueeze(0))
        mod_layers = mod_out["layers_rgba"][0]
        x_hat = mod_out["composited"][0]  # [T, 3, H, W]

        # 5) Apply replay-channel corruption if supplied.
        if self.corruption_fn is not None:
            x_hat = self.corruption_fn(x_hat)

        # 6) Run audits.
        audits: Dict[str, Optional[bool]] = {
            "leakage": None,
            "retrieval": None,
            "locality": None,
        }

        if self.locality is not None:
            passes_loc, _ = self.locality.check(
                src_layers, mod_layers, record.slot_indices
            )
            audits["locality"] = bool(passes_loc)

        if self.leakage_cls is not None and dino_feat_fn is not None:
            # Observe one sample (x_hat at a middle frame as representative).
            t_probe = T // 2
            with torch.no_grad():
                feat = dino_feat_fn(x_hat[t_probe])  # [N, D] or [B=1, N, D]
            self.leakage_cls.observe(feat, family)
            audits["leakage"] = self.leakage_cls.passes_audit()

        if self.retrieval is not None and source_bank_crops is not None:
            # Use mid-frame crops of manipulated layers as queries.
            t_probe = T // 2
            crops = []
            for k in record.slot_indices:
                alpha_k = mod_layers[t_probe, k, 3]
                rgb_k = mod_layers[t_probe, k, :3]
                crops.append(rgb_k * alpha_k.unsqueeze(0))
            audits["retrieval"] = bool(
                self.retrieval.passes_audit(crops, source_bank_crops)
            )

        n_pass = sum(1 for v in audits.values() if v is True)
        # When an audit is None (not configured), we do not count it as pass
        # or fail, but we require at least ``min_audits_passed`` configured
        # passes to emit.
        if n_pass < self.min_audits_passed:
            return None

        event_flags = self._event_flags(record, T=T, K=K)

        return {
            "x_hat": x_hat,
            "replay_labels_for_dustbin": {
                "family": record.family,
                "slot_indices": record.slot_indices,
                "source_K": K,
                "recovered_K_hint": K + (1 if record.family == "birth" else 0)
                                     - (1 if record.family == "kill" else 0),
            },
            "event_flags_for_open_set_head": event_flags,
            "intervention_metadata": record,
            "audits": audits,
        }

    # -------- family implementations --------------------------------------
    def _apply_family(
        self,
        source_slots: torch.Tensor,
        family: str,
        dist_pairs: List[Tuple[int, int]],
        rng: random.Random,
    ) -> Tuple[torch.Tensor, InterventionRecord]:
        T, K, D = source_slots.shape
        manip = source_slots.clone()

        if family == "same":
            return manip, InterventionRecord(family="same", slot_indices=())

        if family == "swap":
            if len(dist_pairs) == 0:
                # Should not occur because emit() falls back to "same" earlier;
                # we guard here for direct callers.
                return source_slots.clone(), InterventionRecord(
                    family="same", slot_indices=()
                )
            i, j = rng.choice(dist_pairs)
            # content-trajectory swap: slot i takes j's trajectory, j takes i's.
            tmp = manip[:, i].clone()
            manip[:, i] = manip[:, j]
            manip[:, j] = tmp
            return manip, InterventionRecord(
                family="swap", slot_indices=(i, j), swap_partner=j
            )

        if family == "gap-return":
            k = rng.randrange(K)
            t_s = rng.randrange(0, max(1, T - 1))
            gap_len = rng.randrange(1, max(2, T - t_s))
            t_e = min(T - 1, t_s + gap_len - 1)
            # Zero the slot during [t_s, t_e]; reappears at t_e+1 via original.
            # We implement "zeroed" as slot vector -> zero; renderer's alpha
            # should collapse. The "return" is automatic because frames after
            # t_e restore the original slot.
            manip[t_s : t_e + 1, k] = 0.0
            return manip, InterventionRecord(
                family="gap-return",
                slot_indices=(k,),
                gap_window=(t_s, t_e),
            )

        if family == "kill":
            k = rng.randrange(K)
            t_kill = rng.randrange(1, max(2, T))
            manip[t_kill:, k] = 0.0
            return manip, InterventionRecord(
                family="kill",
                slot_indices=(k,),
                kill_time=t_kill,
            )

        if family == "birth":
            if self.donor_bank is None or len(self.donor_bank) == 0:
                raise RuntimeError(
                    "birth intervention requested but DonorBank is empty / "
                    "not configured. No silent fallback: the caller must "
                    "pre-warm the donor bank before the first birth event."
                )
            donor = self.donor_bank.sample(rng=rng)
            donor_traj = donor.slot_trajectory.to(source_slots.device).to(source_slots.dtype)
            # Donor traj has shape [T_d, slot_dim]; align length with source.
            T_d = donor_traj.shape[0]
            if T_d >= T:
                donor_traj = donor_traj[:T]
            else:
                # Pad by repeating last frame.
                pad = donor_traj[-1:].repeat(T - T_d, 1)
                donor_traj = torch.cat([donor_traj, pad], dim=0)
            # Choose a free slot index: pick the slot with the lowest
            # time-averaged norm (i.e. most "quiet").
            norms = source_slots.norm(dim=-1).mean(dim=0)  # [K]
            k = int(norms.argmin().item())
            t_birth = rng.randrange(0, max(1, T - 1))
            manip[:, k] = 0.0
            manip[t_birth:, k] = donor_traj[t_birth:]
            return manip, InterventionRecord(
                family="birth",
                slot_indices=(k,),
                birth_time=t_birth,
                donor_id=donor.origin_clip_id,
            )

        raise ValueError(f"Unknown intervention family: {family}")

    # -------- event-flag construction -------------------------------------
    @staticmethod
    def _event_flags(
        record: InterventionRecord, T: int, K: int
    ) -> Dict[str, torch.Tensor]:
        """Produce per-frame per-slot one-hot over {source, dormant, null, born}.

        Channel ordering is driven by the shared ``EVENT_TO_INDEX`` constant
        (module-level, above). This keeps the sampler, the collator, and the
        consumer loss (``GSRSIdentityLoss``) in strict agreement on channel
        identities — the loss converts these one-hots to int labels via
        ``argmax`` and looks up softmax columns in ``open_set_probs`` using
        that same mapping.
        """
        # Default everyone to source.
        flags = torch.zeros(T, K, len(EVENT_TO_INDEX), dtype=torch.float32)
        flags[..., EVENT_TO_INDEX["source"]] = 1.0

        def _set(t0: int, t1: int, k: int, channel: int) -> None:
            flags[t0:t1, k, :] = 0.0
            flags[t0:t1, k, channel] = 1.0

        if record.family == "same":
            pass
        elif record.family == "swap":
            pass  # both swapped slots remain "source" per open-set head semantics
        elif record.family == "gap-return":
            assert record.gap_window is not None
            k = record.slot_indices[0]
            t_s, t_e = record.gap_window
            # During the gap: dormant (temporarily occluded, re-ID'd on return).
            _set(t_s, t_e + 1, k, channel=EVENT_TO_INDEX["dormant"])
        elif record.family == "kill":
            assert record.kill_time is not None
            k = record.slot_indices[0]
            _set(record.kill_time, T, k, channel=EVENT_TO_INDEX["null"])
        elif record.family == "birth":
            assert record.birth_time is not None
            k = record.slot_indices[0]
            # Before birth: null; after: born.
            _set(0, record.birth_time, k, channel=EVENT_TO_INDEX["null"])
            _set(record.birth_time, T, k, channel=EVENT_TO_INDEX["born"])
        return {"per_frame_per_slot": flags}


# --------------------------------------------------------------------------- #
#  Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from slotcontrast.modules.renderer import CompositionalSlotRenderer

    torch.manual_seed(0)
    random.seed(0)
    D, K, T = 64, 5, 6
    renderer = CompositionalSlotRenderer(slot_dim=D, out_size=64, c_init=16, grid_init=8)
    # Reduce out_size / c_init so the smoke test runs fast on CPU.
    print(f"[smoke] small renderer params: {renderer.count_parameters():,}")

    slots = torch.randn(T, K, D)

    # DonorBank
    bank = DonorBank(capacity=8)
    for i in range(3):
        bank.push(DonorEntry(
            slot_trajectory=torch.randn(T, D),
            appearance_crop=torch.zeros(3, 32, 32),
            origin_clip_id=f"clip{i}",
        ))
    print(f"[smoke] donor bank size: {len(bank)}")

    # DistinguishabilityFilter
    df = DistinguishabilityFilter()
    layers = renderer(slots.unsqueeze(0))["layers_rgba"][0]
    pairs = df.distinguishable_pairs(slots, layers)
    print(f"[smoke] distinguishable pairs: {pairs}")

    # LocalityCheck
    loc = LocalityCheck(iou_tolerance=0.1)

    # LeakageClassifier (smoke only)
    lc = LeakageClassifier(dino_feat_dim=768, buffer_size=64)

    # CrossFeatureRetrievalAgreement (smoke with two random-projection "features")
    def _f1(x):
        # [3, H, W] -> D
        return x.mean(dim=(-1, -2))
    def _f2(x):
        return x.mean(dim=(-1, -2)) + 0.01 * torch.randn(3)
    cfa = CrossFeatureRetrievalAgreement(f1=_f1, f2=_f2)

    sampler = InterventionSampler(
        renderer=renderer,
        distinguishability_filter=df,
        donor_bank=bank,
        leakage_classifier=lc,
        retrieval_agreement=cfa,
        locality_check=loc,
        min_audits_passed=0,  # smoke test: allow emission regardless
    )

    # Dummy dino feat fn: project flatten mean to 768.
    def _dino_feat_fn(rgb):
        # rgb [3, H, W] -> [n_patches=4, 768]
        return torch.randn(4, 768)

    source_bank = [torch.rand(3, 64, 64) for _ in range(3)]
    for i in range(6):
        sample = sampler.emit(
            slots, rng=random.Random(i),
            dino_feat_fn=_dino_feat_fn,
            source_bank_crops=source_bank,
        )
        if sample is None:
            print(f"[smoke] iter {i}: audits failed -> None")
        else:
            rec = sample["intervention_metadata"]
            print(f"[smoke] iter {i}: family={rec.family} slots={rec.slot_indices} "
                  f"audits={sample['audits']}")

    # Shape checks for a single forced emission.
    sample = sampler.emit(slots, rng=random.Random(0),
                          dino_feat_fn=_dino_feat_fn,
                          source_bank_crops=source_bank)
    assert sample is not None
    assert sample["x_hat"].shape == (T, 3, 64, 64), sample["x_hat"].shape
    assert sample["event_flags_for_open_set_head"]["per_frame_per_slot"].shape == (T, K, 4)
    print("[smoke] all replay shapes OK")
