"""MOVi-D oracle replay-validity gate for GSRS (§3.5 of
``refine-logs/FINAL_PROPOSAL_2026_04_15.md``).

Implements the four-step protocol:

1. **Trusted source clips**. A source slot is "trusted" iff:
    - slot-to-GT-object purity >= 0.9,
    - visible-mask IoU with its matched GT object >= 0.6,
    - identity stable for >= 80% of frames.

2. **Oracle target event construction** from GT object trajectories, for
   each of the five intervention families.

3. **Renderer-side oracle masks** -- per-layer alpha masks + per-layer crops
   for every frame, produced by the (already-trained) renderer on the
   manipulated slot trajectories.

4. **Validity checks**: geometry, appearance, locality.

Per-family pass-rate targets (Gate G1(d) of §3.8):
    same       >= 0.85
    swap       >= 0.75
    gap-return >= 0.75
    kill       >= 0.80
    birth      >= 0.70

This module consumes **GT masks** at evaluation time (MOVi-D has them);
it is **not** used during main training, only during renderer pretraining
certification and as the periodic audit during full training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
#  Targets                                                                    #
# --------------------------------------------------------------------------- #

PER_FAMILY_TARGETS: Dict[str, float] = {
    "same": 0.85,
    "swap": 0.75,
    "gap-return": 0.75,
    "kill": 0.80,
    "birth": 0.70,
}


# --------------------------------------------------------------------------- #
#  Oracle target event                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class OracleEvent:
    """Per-clip oracle description of the intended event."""

    family: str
    trusted_slots: Tuple[int, ...]
    # For swap:
    swap_pair: Optional[Tuple[int, int]] = None
    # For gap-return / kill:
    gap_window: Optional[Tuple[int, int]] = None
    kill_time: Optional[int] = None
    # For birth:
    birth_time: Optional[int] = None
    birth_slot: Optional[int] = None
    # GT ids that the trusted slots map to.
    gt_object_ids: Tuple[int, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
#  Trust assessment                                                           #
# --------------------------------------------------------------------------- #


class TrustedSlotAssessor:
    """Compute per-slot "trust" based on MOVi-D GT masks (§3.5 Step 1)."""

    def __init__(
        self,
        purity_thresh: float = 0.9,
        iou_thresh: float = 0.6,
        identity_stable_frac: float = 0.8,
    ):
        self.purity_thresh = float(purity_thresh)
        self.iou_thresh = float(iou_thresh)
        self.identity_stable_frac = float(identity_stable_frac)

    @staticmethod
    def _iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """IoU on boolean masks with matching shape [..., H, W]."""
        af = a.float()
        bf = b.float()
        inter = (af * bf).sum(dim=(-1, -2))
        union = (af + bf).clamp(max=1.0).sum(dim=(-1, -2))
        return inter / union.clamp(min=1e-6)

    def trusted_slots(
        self,
        slot_masks: torch.Tensor,  # [T, K, H, W] soft in [0, 1]
        gt_masks: torch.Tensor,     # [T, G, H, W] binary
        binarise_thresh: float = 0.2,
    ) -> Tuple[List[int], Dict[int, int]]:
        """Return (trusted slot indices, slot_to_gt_id map)."""
        if slot_masks.ndim != 4 or gt_masks.ndim != 4:
            raise ValueError(
                "expected slot_masks [T, K, H, W] and gt_masks [T, G, H, W]"
            )
        T, K = slot_masks.shape[:2]
        G = gt_masks.shape[1]
        sm_bin = (slot_masks > binarise_thresh).float()
        gm = gt_masks.float()

        # Per-frame IoU between each slot and each gt.
        # [T, K, G]
        sm_b = sm_bin.unsqueeze(2)  # [T, K, 1, H, W]
        gm_b = gm.unsqueeze(1)      # [T, 1, G, H, W]
        inter = (sm_b * gm_b).sum(dim=(-1, -2))
        union = (sm_b + gm_b).clamp(max=1.0).sum(dim=(-1, -2))
        iou_tkg = inter / union.clamp(min=1e-6)

        # Per-frame argmax GT for each slot.
        per_frame_argmax = iou_tkg.argmax(dim=-1)  # [T, K]
        per_frame_max = iou_tkg.max(dim=-1).values  # [T, K]

        trusted: List[int] = []
        mapping: Dict[int, int] = {}
        for k in range(K):
            # Most frequent gt across frames.
            am = per_frame_argmax[:, k]
            if am.numel() == 0:
                continue
            u = torch.unique(am, return_counts=True)
            top_gt = int(u[0][u[1].argmax()].item())
            stable_frac = (am == top_gt).float().mean().item()
            if stable_frac < self.identity_stable_frac:
                continue
            # Mean IoU in frames where slot matches top_gt.
            frames_match = (am == top_gt)
            if not frames_match.any():
                continue
            mean_iou = per_frame_max[frames_match, k].mean().item()
            if mean_iou < self.iou_thresh:
                continue
            # Purity: fraction of slot's alpha mass that falls on top_gt's
            # GT mask, averaged across frames.
            gt_mask_top = gm[:, top_gt]  # [T, H, W]
            slot_mass = sm_bin[:, k].sum(dim=(-1, -2)).clamp(min=1e-6)
            inside = (sm_bin[:, k] * gt_mask_top).sum(dim=(-1, -2))
            purity_per_frame = inside / slot_mass
            if purity_per_frame.mean().item() < self.purity_thresh:
                continue
            trusted.append(k)
            mapping[k] = top_gt
        return trusted, mapping


# --------------------------------------------------------------------------- #
#  Validity checks                                                            #
# --------------------------------------------------------------------------- #


class ValidityChecker:
    """Implements §3.5 Step 4 validity predicates."""

    def __init__(
        self,
        geom_iou_min: float = 0.55,
        timing_tolerance_frames: int = 1,
        active_count_exact_frac: float = 0.95,
        appearance_margin: float = 0.05,
        locality_iou_tolerance: float = 0.1,
        alpha_bin_thresh: float = 0.2,
    ):
        self.geom_iou_min = float(geom_iou_min)
        self.timing_tolerance_frames = int(timing_tolerance_frames)
        self.active_count_exact_frac = float(active_count_exact_frac)
        self.appearance_margin = float(appearance_margin)
        self.locality_iou_tolerance = float(locality_iou_tolerance)
        self.alpha_bin_thresh = float(alpha_bin_thresh)

    # -------- geometry -----------------------------------------------------
    def geometry(
        self,
        rendered_alpha: torch.Tensor,  # [T, H, W] soft
        oracle_target: torch.Tensor,    # [T, H, W] binary
        event_frame: Optional[int] = None,
        event_frame_pred: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a dict with ``pass``, ``iou``, ``timing_off``,
        ``active_exact_frac``."""
        if rendered_alpha.shape != oracle_target.shape:
            raise ValueError(
                f"shape mismatch rendered {tuple(rendered_alpha.shape)} vs "
                f"oracle {tuple(oracle_target.shape)}"
            )
        ra_bin = (rendered_alpha > self.alpha_bin_thresh).float()
        ot = oracle_target.float()
        inter = (ra_bin * ot).sum(dim=(-1, -2))
        union = (ra_bin + ot).clamp(max=1.0).sum(dim=(-1, -2))
        iou_per_frame = inter / union.clamp(min=1e-6)
        mean_iou = iou_per_frame.mean().item()

        active_pred = ra_bin.sum(dim=(-1, -2)) > 0  # [T]
        active_tgt = ot.sum(dim=(-1, -2)) > 0
        active_exact_frac = (active_pred == active_tgt).float().mean().item()

        timing_off: Optional[int] = None
        if event_frame is not None and event_frame_pred is not None:
            timing_off = int(abs(event_frame - event_frame_pred))

        passes = (
            mean_iou >= self.geom_iou_min
            and active_exact_frac >= self.active_count_exact_frac
            and (timing_off is None
                 or timing_off <= self.timing_tolerance_frames)
        )
        return {
            "pass": bool(passes),
            "iou": mean_iou,
            "active_exact_frac": active_exact_frac,
            "timing_off": timing_off,
        }

    # -------- appearance ---------------------------------------------------
    def appearance(
        self,
        rendered_embed: torch.Tensor,  # [D] embedding of rendered crop
        source_embed: torch.Tensor,     # [D] embedding of source crop
        competitor_embeds: torch.Tensor,  # [M, D] embeddings of competing manipulated objects
    ) -> Dict[str, Any]:
        """Return pass iff cos(rendered, source) exceeds max cos(rendered,
        competitor) by at least ``self.appearance_margin``."""
        r = F.normalize(rendered_embed, dim=-1)
        s = F.normalize(source_embed, dim=-1)
        if competitor_embeds.numel() == 0:
            # No competitors -> trivially passes.
            margin = 1.0
        else:
            c = F.normalize(competitor_embeds, dim=-1)
            sim_src = (r * s).sum(dim=-1).item()
            sim_comp = (c @ r).max().item()
            margin = sim_src - sim_comp
        return {
            "pass": bool(margin >= self.appearance_margin),
            "margin": margin,
        }

    # -------- locality -----------------------------------------------------
    def locality(
        self,
        src_layers_rgba: torch.Tensor,  # [T, K, 4, H, W]
        mod_layers_rgba: torch.Tensor,
        manipulated_slot_indices: Sequence[int],
    ) -> Dict[str, Any]:
        if src_layers_rgba.shape != mod_layers_rgba.shape:
            raise ValueError("source/manipulated layer shapes must match")
        T, K = src_layers_rgba.shape[:2]
        sa = (src_layers_rgba[:, :, 3] > self.alpha_bin_thresh).float()
        ma = (mod_layers_rgba[:, :, 3] > self.alpha_bin_thresh).float()
        manip = set(manipulated_slot_indices)
        worst_delta = 0.0
        per_slot: List[float] = []
        for k in range(K):
            if k in manip:
                per_slot.append(float("nan"))
                continue
            inter = (sa[:, k] * ma[:, k]).sum(dim=(-1, -2))
            union = (sa[:, k] + ma[:, k]).clamp(max=1.0).sum(dim=(-1, -2))
            iou = (inter / union.clamp(min=1e-6)).mean().item()
            per_slot.append(iou)
            worst_delta = max(worst_delta, 1.0 - iou)
        return {
            "pass": bool(worst_delta <= self.locality_iou_tolerance),
            "worst_delta": worst_delta,
            "per_slot_iou": per_slot,
        }


# --------------------------------------------------------------------------- #
#  Oracle event construction                                                  #
# --------------------------------------------------------------------------- #


def _gt_trajectory_mask(
    gt_masks: torch.Tensor, gt_id: int
) -> torch.Tensor:
    """gt_masks [T, G, H, W] -> [T, H, W] for the chosen id."""
    return gt_masks[:, gt_id]


def build_oracle_event(
    family: str,
    trusted_slots: Sequence[int],
    slot_to_gt: Dict[int, int],
    T: int,
    rng,
    gap_len_range: Tuple[int, int] = (2, 6),
) -> Tuple[OracleEvent, Dict[int, torch.Tensor]]:
    """Produce an ``OracleEvent`` and a dict of per-trusted-slot target masks
    ``[T, H, W]``. The caller supplies GT masks via ``oracle_masks_lookup``
    after this call; here we return the *intended* mask-selection rules.

    For the smoke-testable flow, we return just the OracleEvent; the actual
    oracle masks are assembled by ``MOViDOracleValidator.validate_clip``.
    """
    if not trusted_slots:
        raise RuntimeError(
            "build_oracle_event called with zero trusted slots; the caller "
            "should screen these clips out upstream."
        )

    event = OracleEvent(
        family=family,
        trusted_slots=tuple(trusted_slots),
        gt_object_ids=tuple(slot_to_gt[k] for k in trusted_slots),
    )

    if family == "same":
        pass
    elif family == "swap":
        if len(trusted_slots) < 2:
            raise RuntimeError(
                "swap family requires >=2 trusted slots; caller should "
                "screen out clips that don't qualify."
            )
        # Choose the first two trusted slots (deterministic for testing).
        event.swap_pair = (trusted_slots[0], trusted_slots[1])
    elif family == "gap-return":
        k = trusted_slots[0]
        t_s = rng.randrange(1, max(2, T - gap_len_range[1]))
        g = rng.randrange(gap_len_range[0], gap_len_range[1] + 1)
        t_e = min(T - 2, t_s + g - 1)
        event.gap_window = (t_s, t_e)
    elif family == "kill":
        t_kill = rng.randrange(1, max(2, T))
        event.kill_time = t_kill
    elif family == "birth":
        t_birth = rng.randrange(1, max(2, T - 1))
        event.birth_time = t_birth
        # The birth slot is assigned by the renderer / sampler; here we use
        # a placeholder (None -> caller fills).
        event.birth_slot = None
    else:
        raise ValueError(f"Unknown intervention family: {family}")

    return event, {}


# --------------------------------------------------------------------------- #
#  MOViDOracleValidator                                                       #
# --------------------------------------------------------------------------- #


class MOViDOracleValidator:
    """Top-level MOVi-D oracle replay-validity validator.

    Contract (forward):
        ``validate_batch(batch)`` where ``batch`` is a dict with:

        - ``slot_trajectories``: [B, T, K, slot_dim] from the frozen teacher.
        - ``gt_masks``:           [B, T, G, H, W] binary.
        - ``renderer_out_source``: dict from ``renderer(slot_trajectories)``
          containing ``layers_rgba`` and ``composited``.
        - ``renderer_out_mod``:    dict from ``renderer(manipulated_slots)``.
        - ``manipulated_slot_indices``: per-batch list of which slots were
          touched.
        - ``family``:              per-batch intervention family label.
        - ``appearance_embeds``:   dict with:
              'rendered':    [B, len(manip), D] crop embeds from mod,
              'source':      [B, len(manip), D] crop embeds from src,
              'competitors': [B, len(manip), M, D] other manipulated embeds.
        - ``event_timing``:        dict with optional 'event_frame',
              'event_frame_pred' per sample.

    Returns:
        {
            'pass_rate':       {family: float},
            'pass_mask':       [B] bool,
            'per_family_stats': {family: {'n': int, 'n_pass': int, ...}},
            'per_sample_detail': list of dicts (length B),
            'per_family_targets': PER_FAMILY_TARGETS,
            'targets_met':     {family: bool},
        }
    """

    def __init__(
        self,
        trusted: Optional[TrustedSlotAssessor] = None,
        checker: Optional[ValidityChecker] = None,
    ):
        self.trusted = trusted or TrustedSlotAssessor()
        self.checker = checker or ValidityChecker()

    # -------- per-clip validation -----------------------------------------
    def _validate_one(
        self,
        family: str,
        src_layers: torch.Tensor,     # [T, K, 4, H, W]
        mod_layers: torch.Tensor,
        manipulated_slot_indices: Sequence[int],
        oracle_target_per_manip: Sequence[torch.Tensor],  # list of [T, H, W]
        appearance: Optional[Dict[str, torch.Tensor]],
        event_timing: Optional[Dict[str, int]],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"family": family, "sub_checks": {}}

        # Geometry: aggregate across manipulated slots (if any).
        if family == "same":
            # No manipulated slots; check that ALL layers match their source
            # rendering, i.e. locality over all slots.
            passes, deltas = self.checker.locality(
                src_layers, mod_layers, manipulated_slot_indices=[]
            )["pass"], None
            # For "same", also use geometry on a representative slot if
            # oracle_target is non-empty.
            geom_passes = []
            if oracle_target_per_manip:
                for oracle_t, k in zip(oracle_target_per_manip, manipulated_slot_indices):
                    g = self.checker.geometry(mod_layers[:, k, 3], oracle_t)
                    geom_passes.append(g["pass"])
                    result["sub_checks"].setdefault("geometry", []).append(g)
            result["sub_checks"]["locality"] = {"pass": passes}
            result["pass"] = bool(passes and all(geom_passes) if geom_passes else passes)
            return result

        # Geometry per manipulated slot.
        geom_passes = []
        for oracle_t, k in zip(oracle_target_per_manip, manipulated_slot_indices):
            ef, efp = None, None
            if event_timing is not None:
                ef = event_timing.get("event_frame")
                efp = event_timing.get("event_frame_pred")
            g = self.checker.geometry(mod_layers[:, k, 3], oracle_t,
                                      event_frame=ef, event_frame_pred=efp)
            geom_passes.append(g["pass"])
            result["sub_checks"].setdefault("geometry", []).append(g)

        # Locality on untouched slots.
        loc = self.checker.locality(src_layers, mod_layers, manipulated_slot_indices)
        result["sub_checks"]["locality"] = loc

        # Appearance (optional).
        app_passes = True
        if appearance is not None:
            rendered = appearance.get("rendered")
            source = appearance.get("source")
            competitors = appearance.get("competitors")
            if rendered is not None and source is not None and competitors is not None:
                app_all = []
                n = rendered.shape[0]
                for i in range(n):
                    a = self.checker.appearance(
                        rendered[i], source[i], competitors[i]
                    )
                    app_all.append(a["pass"])
                    result["sub_checks"].setdefault("appearance", []).append(a)
                app_passes = all(app_all)

        result["pass"] = bool(all(geom_passes) and loc["pass"] and app_passes)
        return result

    # -------- batch entry --------------------------------------------------
    def validate_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        required = [
            "renderer_out_source", "renderer_out_mod",
            "manipulated_slot_indices", "family",
        ]
        for key in required:
            if key not in batch:
                raise KeyError(f"validate_batch: missing '{key}' in batch dict")

        src_out = batch["renderer_out_source"]
        mod_out = batch["renderer_out_mod"]
        if "layers_rgba" not in src_out or "layers_rgba" not in mod_out:
            raise KeyError(
                "renderer_out_source and renderer_out_mod must contain "
                "'layers_rgba' (set return_layers=True when calling renderer)."
            )
        src_layers = src_out["layers_rgba"]  # [B, T, K, 4, H, W]
        mod_layers = mod_out["layers_rgba"]
        B = src_layers.shape[0]

        per_sample: List[Dict[str, Any]] = []
        pass_mask = torch.zeros(B, dtype=torch.bool)
        per_family_counts: Dict[str, Dict[str, int]] = {
            f: {"n": 0, "n_pass": 0} for f in PER_FAMILY_TARGETS
        }

        for b in range(B):
            family = batch["family"][b]
            manip_idx = batch["manipulated_slot_indices"][b]
            oracle_targets = batch.get("oracle_target_per_manip", [[]] * B)[b]
            appearance = None
            if "appearance_embeds" in batch:
                ae = batch["appearance_embeds"]
                appearance = {
                    "rendered": ae["rendered"][b],
                    "source": ae["source"][b],
                    "competitors": ae["competitors"][b],
                }
            event_timing = None
            if "event_timing" in batch:
                event_timing = batch["event_timing"][b]

            detail = self._validate_one(
                family=family,
                src_layers=src_layers[b],
                mod_layers=mod_layers[b],
                manipulated_slot_indices=manip_idx,
                oracle_target_per_manip=oracle_targets,
                appearance=appearance,
                event_timing=event_timing,
            )
            per_sample.append(detail)
            pass_mask[b] = detail["pass"]
            per_family_counts.setdefault(family, {"n": 0, "n_pass": 0})
            per_family_counts[family]["n"] += 1
            if detail["pass"]:
                per_family_counts[family]["n_pass"] += 1

        pass_rate: Dict[str, float] = {}
        per_family_stats: Dict[str, Dict[str, Any]] = {}
        targets_met: Dict[str, bool] = {}
        for fam, counts in per_family_counts.items():
            if counts["n"] == 0:
                pass_rate[fam] = float("nan")
                targets_met[fam] = False
            else:
                r = counts["n_pass"] / counts["n"]
                pass_rate[fam] = r
                targets_met[fam] = r >= PER_FAMILY_TARGETS.get(fam, 0.0)
            per_family_stats[fam] = {
                "n": counts["n"],
                "n_pass": counts["n_pass"],
                "rate": pass_rate[fam],
                "target": PER_FAMILY_TARGETS.get(fam, float("nan")),
            }

        return {
            "pass_rate": pass_rate,
            "pass_mask": pass_mask,
            "per_family_stats": per_family_stats,
            "per_sample_detail": per_sample,
            "per_family_targets": dict(PER_FAMILY_TARGETS),
            "targets_met": targets_met,
        }


# --------------------------------------------------------------------------- #
#  Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import random as _random

    import importlib.util
    import os
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = importlib.util.spec_from_file_location(
        "renderer", os.path.join(repo, "slotcontrast/modules/renderer.py")
    )
    rend_mod = importlib.util.module_from_spec(spec)
    sys.modules["renderer"] = rend_mod
    spec.loader.exec_module(rend_mod)
    Renderer = rend_mod.CompositionalSlotRenderer

    torch.manual_seed(0)
    _random.seed(0)
    B, T, K, D, G, Hw = 3, 5, 4, 64, 3, 64
    renderer = Renderer(slot_dim=D, out_size=Hw, c_init=16, grid_init=8)

    slots = torch.randn(B, T, K, D)
    src = renderer(slots)
    mod = renderer(slots + 0.1 * torch.randn_like(slots))

    # Fake GT masks: blob per frame per object.
    gt = torch.zeros(B, T, G, Hw, Hw)
    for b in range(B):
        for t in range(T):
            for g in range(G):
                yy, xx = torch.meshgrid(
                    torch.arange(Hw), torch.arange(Hw), indexing="ij"
                )
                cy = Hw // 2 + 5 * g
                cx = Hw // 2 - 5 * g
                gt[b, t, g] = ((yy - cy) ** 2 + (xx - cx) ** 2 < 100).float()

    # Trust assessor (likely rejects given random slots; smoke tests API).
    ta = TrustedSlotAssessor()
    for b in range(B):
        trusted, mapping = ta.trusted_slots(
            slot_masks=src["layers_rgba"][b][:, :, 3], gt_masks=gt[b]
        )
        print(f"[smoke] clip {b} trusted slots: {trusted}  mapping: {mapping}")

    # Build a batch for validate_batch.
    families = ["same", "kill", "gap-return"]
    manipulated_slot_indices = [[], [0], [1]]
    oracle_targets = [
        [],                      # same
        [gt[1, :, 0]],           # kill -- target = GT id 0 trajectory
        [gt[2, :, 1]],           # gap-return -- target = GT id 1 trajectory
    ]
    batch = {
        "renderer_out_source": {"layers_rgba": src["layers_rgba"]},
        "renderer_out_mod":    {"layers_rgba": mod["layers_rgba"]},
        "manipulated_slot_indices": manipulated_slot_indices,
        "family": families,
        "oracle_target_per_manip": oracle_targets,
    }
    validator = MOViDOracleValidator()
    out = validator.validate_batch(batch)
    print(f"[smoke] pass_mask:       {out['pass_mask'].tolist()}")
    print(f"[smoke] pass_rate:       {out['pass_rate']}")
    print(f"[smoke] per_family_stats:{out['per_family_stats']}")
    print(f"[smoke] targets_met:     {out['targets_met']}")
    assert out["pass_mask"].shape == (B,)
    print("[smoke] validator shapes OK")

    # build_oracle_event
    rng = _random.Random(0)
    for fam in ("same", "gap-return", "kill", "birth"):
        ev, _ = build_oracle_event(
            family=fam, trusted_slots=[0], slot_to_gt={0: 0}, T=T, rng=rng
        )
        print(f"[smoke] oracle event {fam}: {ev}")
    print("[smoke] all replay_validity smoke checks passed.")
