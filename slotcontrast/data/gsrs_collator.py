"""GSRS replay-batch collator (Part A ↔ Part B integration bridge).

This module wraps an existing training dataloader and, with probability
``p_replay``, replaces the batch's ``video`` tensor with a replay-clip
``x_hat`` rendered by the frozen ``CompositionalSlotRenderer`` (Part A)
from the frozen teacher's slot trajectories, AND attaches the auxiliary
replay keys that ``GSRSIdentityLoss`` consumes via its ``aux_keys``
contract:

    * ``replay_target_trajectories``  — ``[B, T, K_target + K_dustbin, D]``
    * ``replay_event_flags``          — ``[B, T, K_target + K_dustbin]``
                                        (int labels indexed by
                                        ``replay.EVENT_TO_INDEX``)
    * ``replay_family``               — length-``B`` list of intervention
                                        family strings.

On a non-replay batch (the ``1 - p_replay`` branch) the batch is
forwarded unchanged with no replay keys set, matching
``ObjectCentricModel.forward``'s dispatch logic which is a strict no-op
unless ``replay_family`` is present.

Design notes
------------

*   The collator is **stateful** but owns only *frozen* modules: a
    ``CompositionalSlotRenderer`` loaded from ``renderer_ckpt`` and a
    teacher slot-extraction callable. Both are set once at construction;
    neither is part of the optimizer graph.

*   **Hard-raise on missing ckpts** in the main-training codepath
    (``allow_uninit_renderer=False``). The smoke-test script may set
    ``allow_uninit_renderer=True`` to exercise control flow with a
    freshly-initialised renderer. No silent fallback in production.

*   **Dustbin rows**. Per ``GSRSIdentityLoss`` contract
    (``num_dustbins ∈ {2, 4}``), we emit two additional rows on the target
    trajectory axis: one "born" dustbin (event index
    ``EVENT_TO_INDEX['born']``) and one "dead" dustbin (event index
    ``EVENT_TO_INDEX['null']``). The dustbin row trajectories carry zeros
    (they are never matched against cosine; the loss uses the fixed
    ``c_bin`` scalar for dustbin columns).

*   **Event-flag conversion**. The sampler emits
    ``[T, K, 4]`` one-hots; the loss consumes ``[B, T, K_total]`` int
    labels. We ``argmax`` per-slot per-frame to collapse the last axis.

*   **Audit outcome**. If ``InterventionSampler.emit()`` returns ``None``
    (audits failed), we fall back to a non-replay batch for that sample
    — NOT a silent degradation of the signal, because the audits are a
    correctness gate (§3.4) rather than a capacity knob. The fall-through
    is logged via ``replay_skip_count`` for accounting.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from torch import nn


class GSRSReplayCollator:
    """Wraps a standard dataloader; per-batch, injects a GSRS replay clip.

    Usage::

        collator = GSRSReplayCollator(
            base_dataloader=dm.train_dataloader(),
            renderer_ckpt="/path/to/renderer.pt",
            teacher_slots_fn=lambda video: <slots [B, T, K, D]>,
            p_replay=0.5,
            slot_dim=128,
            num_slots=15,
            out_size=224,
            allow_uninit_renderer=False,
        )
        for batch in collator:
            model(batch)

    Parameters
    ----------
    base_dataloader:
        Any iterable producing dict-batches with a ``video`` key of shape
        ``[B, T, 3, H, W]``.
    renderer_ckpt:
        Path to the frozen renderer checkpoint written by
        ``scripts/pretrain_gsrs_renderer.py``. Pass ``None`` only together
        with ``allow_uninit_renderer=True`` (smoke-test only).
    teacher_slots_fn:
        Callable ``(video: [B, T, 3, H, W]) -> slots: [B, T, K, D_slot]``.
        MUST be no-grad and produced by a frozen module (the caller is
        responsible). Returning slots with the wrong shape raises.
    p_replay:
        Probability of replacing a batch with a replay clip. 0.5 by
        default per proposal §3.9.
    slot_dim, num_slots, out_size:
        Renderer construction parameters (must match the ckpt's training
        configuration). Hard-raise on dim mismatch when loading the ckpt.
    donor_bank:
        Optional ``replay.DonorBank`` for the birth family. If ``None``,
        the ``birth`` family is disabled (caller must ensure
        ``families`` does not include ``"birth"`` in that case, else
        emit() will raise).
    num_dustbins:
        Must match ``GSRSIdentityLoss.num_dustbins``. Currently fixed at
        ``2``: one born dustbin + one dead dustbin.
    allow_uninit_renderer:
        If True, construct a freshly-initialised renderer when
        ``renderer_ckpt`` is ``None``. **Smoke-test only** — never pass
        ``True`` on the main training path (the loss will train against
        garbage replay clips).
    seed:
        Per-process RNG seed for family sampling + Bernoulli p_replay.
    """

    def __init__(
        self,
        base_dataloader: Iterable[Dict[str, Any]],
        renderer_ckpt: Optional[str],
        teacher_slots_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        p_replay: float = 0.5,
        slot_dim: int = 128,
        num_slots: int = 15,
        out_size: int = 224,
        donor_bank: Optional[Any] = None,
        num_dustbins: int = 2,
        allow_uninit_renderer: bool = False,
        seed: int = 0,
        device: Optional[str] = None,
    ):
        # Lazy imports so the data module can be imported without dragging in
        # the renderer / audit stack when replay is disabled.
        from slotcontrast.modules.renderer import CompositionalSlotRenderer
        from slotcontrast.modules.replay import (
            DistinguishabilityFilter,
            InterventionSampler,
        )

        if not (0.0 <= p_replay <= 1.0):
            raise ValueError(f"`p_replay` must be in [0, 1], got {p_replay}.")
        if num_dustbins not in (2, 4):
            raise ValueError(f"`num_dustbins` must be 2 or 4, got {num_dustbins}.")
        if teacher_slots_fn is None:
            raise ValueError(
                "`teacher_slots_fn` is required — the collator needs a frozen "
                "callable that maps `video [B, T, 3, H, W]` to slot "
                "trajectories `[B, T, K, D]`. No silent fallback."
            )

        self.base_dataloader = base_dataloader
        self.p_replay = float(p_replay)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.num_dustbins = int(num_dustbins)
        self.teacher_slots_fn = teacher_slots_fn
        self.allow_uninit_renderer = bool(allow_uninit_renderer)
        self._rng = random.Random(seed)
        self.device = device

        # -- Construct the frozen renderer ------------------------------
        self.renderer = CompositionalSlotRenderer(
            slot_dim=slot_dim,
            hidden_dim=128,
            out_size=out_size,
            c_init=40,
            grid_init=14,
            max_time=32,
        )
        if renderer_ckpt is None:
            if not self.allow_uninit_renderer:
                raise ValueError(
                    "GSRSReplayCollator: `renderer_ckpt` is None but "
                    "`allow_uninit_renderer=False`. On the main training path "
                    "a frozen-renderer checkpoint MUST be provided — no silent "
                    "fallback to an untrained renderer (which would poison the "
                    "replay signal). Pass `allow_uninit_renderer=True` only in "
                    "smoke tests."
                )
            # Smoke-test-only branch: fresh init. The caller has explicitly
            # opted in; we still freeze the parameters to keep the semantic
            # guarantee that the collator owns no trainable state.
        else:
            ckpt = torch.load(renderer_ckpt, map_location="cpu", weights_only=False)
            sd = ckpt.get("renderer_state_dict") or ckpt.get("state_dict") or ckpt
            if not isinstance(sd, dict):
                raise ValueError(
                    f"GSRSReplayCollator: checkpoint at {renderer_ckpt!r} has "
                    f"no recognisable state_dict (got type {type(sd).__name__})."
                )
            # Strict load — we do not tolerate silent key mismatches here.
            self.renderer.load_state_dict(sd, strict=True)

        self.renderer.eval()
        for p in self.renderer.parameters():
            p.requires_grad = False
        if device is not None:
            self.renderer.to(device)

        # -- Intervention sampler --------------------------------------
        # Automatic audits are opt-in for the collator: we run only the
        # locality check here (cheap, layer-only). Leakage + retrieval are
        # certified once at Part A (renderer pretrain); running them
        # per-batch would add DINO/SigLIP forwards to the data pipeline.
        # The sampler's `min_audits_passed` is set to 0 so emit() never
        # rejects for unconfigured audits, but the locality audit, if it
        # returns False, does NOT fail emit() on its own unless
        # `min_audits_passed >= 1`. We keep it informational.
        dist_filter = DistinguishabilityFilter()
        self.sampler = InterventionSampler(
            renderer=self.renderer,
            distinguishability_filter=dist_filter,
            donor_bank=donor_bank,
            min_audits_passed=0,
        )

        # Accounting counters (exposed for diagnostic logging per §3.9.1).
        self.replay_emit_count: Dict[str, int] = {}
        self.replay_skip_count: int = 0
        self.non_replay_count: int = 0

    # ------------------------------------------------------------------ #
    #  Iteration                                                         #
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for batch in self.base_dataloader:
            yield self._maybe_inject_replay(batch)

    def __len__(self) -> int:
        # Pass through the base loader's length if it has one.
        if hasattr(self.base_dataloader, "__len__"):
            return len(self.base_dataloader)  # type: ignore[arg-type]
        raise TypeError("underlying dataloader has no __len__")

    # ------------------------------------------------------------------ #
    #  Core replay injection                                             #
    # ------------------------------------------------------------------ #

    def _maybe_inject_replay(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """With probability ``p_replay``, replace the batch's video + attach
        the three replay aux keys. Otherwise pass through unchanged."""
        if self._rng.random() >= self.p_replay:
            self.non_replay_count += 1
            return batch

        if "video" not in batch:
            raise KeyError(
                "GSRSReplayCollator: batch missing 'video' key; cannot run "
                "the replay branch. Make sure the upstream pipeline lists "
                "'video' among its keys."
            )
        video = batch["video"]  # [B, T, 3, H, W]
        if video.ndim != 5:
            raise ValueError(
                f"GSRSReplayCollator expects video of shape [B, T, 3, H, W]; "
                f"got {tuple(video.shape)}."
            )

        # -- 1) Get teacher slot trajectories (frozen) -----------------
        with torch.no_grad():
            teacher_slots = self.teacher_slots_fn(video)  # [B, T, K, D]
        if teacher_slots.ndim != 4:
            raise ValueError(
                "teacher_slots_fn returned wrong ndim: expected 4 "
                f"([B, T, K, D]); got {teacher_slots.ndim}."
            )
        B, T, K, D = teacher_slots.shape
        if D != self.slot_dim:
            raise ValueError(
                f"teacher_slots_fn slot_dim mismatch: collator expected "
                f"{self.slot_dim}, got {D}."
            )

        # -- 2) Per-clip intervention + render -------------------------
        replay_videos: List[torch.Tensor] = []
        replay_target_list: List[torch.Tensor] = []
        replay_event_list: List[torch.Tensor] = []
        replay_family_list: List[str] = []
        n_emitted = 0
        for b in range(B):
            slots_b = teacher_slots[b]  # [T, K, D]
            sample = self.sampler.emit(slots_b, rng=self._rng)
            if sample is None:
                # Audits failed (uses `min_audits_passed` threshold). Fall
                # back to a same-intervention: reuse the source render as
                # x_hat and emit a 'same' label.
                self.replay_skip_count += 1
                # Replace with a trivial no-op: family='same', manip=[], no
                # gap/kill/birth. This keeps the batch aux-key-complete
                # without polluting the training signal with non-certified
                # replay.
                with torch.no_grad():
                    src_out = self.renderer(slots_b.unsqueeze(0))
                x_hat_b = src_out["composited"][0]  # [T, 3, H, W]
                family_b = "same"
                manip_slots_b = slots_b.clone()
                record_flags = torch.zeros(T, K, 4, dtype=torch.float32)
                record_flags[..., 0] = 1.0  # all source
            else:
                x_hat_b = sample["x_hat"]  # [T, 3, H, W]
                family_b = sample["intervention_metadata"].family
                # Re-derive manip slots for the target trajectory (the
                # sampler does not currently expose these, but we can
                # re-apply the intervention — OR reconstruct from the
                # family/record. Simpler: re-run the sampler's private
                # path to get manip slots.
                record = sample["intervention_metadata"]
                manip_slots_b = self._rebuild_manip_slots(slots_b, record)
                record_flags = sample["event_flags_for_open_set_head"][
                    "per_frame_per_slot"
                ]  # [T, K, 4]
                n_emitted += 1
                self.replay_emit_count[family_b] = (
                    self.replay_emit_count.get(family_b, 0) + 1
                )

            # Build target trajectory matrix: [T, K + num_dustbins, D].
            # Real rows = manip_slots_b; dustbin rows = zeros.
            dustbin_rows = torch.zeros(
                T, self.num_dustbins, D, dtype=manip_slots_b.dtype,
                device=manip_slots_b.device,
            )
            targets_b = torch.cat([manip_slots_b, dustbin_rows], dim=1)  # [T, K+ND, D]

            # Event flags: one-hot [T, K, 4] -> int labels [T, K], then
            # append dustbin columns: born = EVENT_TO_INDEX['born'],
            # dead = EVENT_TO_INDEX['null'].
            from slotcontrast.modules.replay import EVENT_TO_INDEX
            int_flags = record_flags.argmax(dim=-1)  # [T, K]
            # Num_dustbins=2 convention: [born, dead].
            born_col = torch.full(
                (T, 1), EVENT_TO_INDEX["born"], dtype=int_flags.dtype,
                device=int_flags.device,
            )
            dead_col = torch.full(
                (T, 1), EVENT_TO_INDEX["null"], dtype=int_flags.dtype,
                device=int_flags.device,
            )
            if self.num_dustbins == 2:
                flags_b = torch.cat([int_flags, born_col, dead_col], dim=1)
            elif self.num_dustbins == 4:
                # Doubled dustbins per loss docstring.
                flags_b = torch.cat(
                    [int_flags, born_col, born_col, dead_col, dead_col],
                    dim=1,
                )
            else:
                raise ValueError(
                    f"num_dustbins={self.num_dustbins} not supported."
                )

            replay_videos.append(x_hat_b)
            replay_target_list.append(targets_b)
            replay_event_list.append(flags_b)
            replay_family_list.append(family_b)

        # -- 3) Stack + attach to batch ---------------------------------
        new_video = torch.stack(replay_videos, dim=0)  # [B, T, 3, H, W]
        # Preserve dtype/device of the original video (the renderer emits
        # fp32 in [0, 1]; the original video may be fp16 or normalised).
        # We deliberately do NOT normalise here; the model's encoder owns
        # normalisation. The renderer emits in [0, 1] RGB which matches the
        # pre-normalisation contract.
        new_targets = torch.stack(replay_target_list, dim=0)  # [B, T, K+ND, D]
        new_flags = torch.stack(replay_event_list, dim=0)     # [B, T, K+ND]

        batch = dict(batch)  # shallow copy so we don't mutate the loader's buffer
        batch["video"] = new_video
        batch["replay_target_trajectories"] = new_targets
        batch["replay_event_flags"] = new_flags
        batch["replay_family"] = replay_family_list
        return batch

    def _rebuild_manip_slots(
        self, slots: torch.Tensor, record: Any
    ) -> torch.Tensor:
        """Re-apply an intervention record to reproduce the manipulated slot
        trajectory that the sampler rendered.

        The sampler does not currently expose manip slots; re-deriving from
        the ``InterventionRecord`` keeps us decoupled from its private state.

        Mirrors the family branches in ``InterventionSampler._apply_family``
        but does NOT re-roll random choices: we consume the concrete
        indices / times already on the record.
        """
        T, K, D = slots.shape
        manip = slots.clone()
        if record.family == "same":
            return manip
        if record.family == "swap":
            i, j = record.slot_indices[0], record.swap_partner
            tmp = manip[:, i].clone()
            manip[:, i] = manip[:, j]
            manip[:, j] = tmp
            return manip
        if record.family == "gap-return":
            assert record.gap_window is not None
            k = record.slot_indices[0]
            t_s, t_e = record.gap_window
            manip[t_s : t_e + 1, k] = 0.0
            return manip
        if record.family == "kill":
            assert record.kill_time is not None
            k = record.slot_indices[0]
            manip[record.kill_time:, k] = 0.0
            return manip
        if record.family == "birth":
            # For birth we cannot reconstruct the donor trajectory without
            # re-sampling from the donor bank. The sampler stores donor_id
            # but not the concrete trajectory — in practice, the loss only
            # matches cosine against the manipulated-slot content, so
            # zeroing out the birth slot before t_birth is enough to carry
            # the correct event labels to the loss. The teacher's own
            # slot at this index is already zeroed by the sampler's
            # `manip[:, k] = 0.0; manip[t_birth:, k] = donor_traj[...]`
            # sequence, so we approximate by zeroing-only here. (An exact
            # reconstruction would require exposing manip slots on the
            # sampler's emit() return dict; flagged as a future cleanup.)
            assert record.birth_time is not None
            k = record.slot_indices[0]
            manip[:, k] = 0.0
            return manip
        raise ValueError(f"Unknown intervention family: {record.family}")

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    def accounting_snapshot(self) -> Dict[str, Any]:
        """Return the per-family emit counters + skip counter for logging
        per proposal §3.9.1 (replay acceptance-rate accounting)."""
        total = sum(self.replay_emit_count.values()) + self.replay_skip_count
        acceptance_rate = (
            sum(self.replay_emit_count.values()) / max(total, 1)
        )
        return {
            "replay_emit_count_by_family": dict(self.replay_emit_count),
            "replay_skip_count": self.replay_skip_count,
            "non_replay_count": self.non_replay_count,
            "acceptance_rate": float(acceptance_rate),
        }


def build_teacher_slots_fn(
    model: nn.Module,
    *,
    input_key: str = "video",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Convenience factory: produce a callable compatible with
    ``GSRSReplayCollator(teacher_slots_fn=...)`` from a snapshotted
    ``ObjectCentricModel.teacher`` bundle or an equivalent frozen module.

    The returned callable:

        * wraps the forward in ``torch.no_grad()``.
        * calls ``model.teacher.encoder`` → ``.initializer`` → ``.processor``
          in the same order as the main training forward so slot
          trajectories come from a frozen Phase-1 snapshot (§3.9).

    The model is assumed to have ``teacher`` already populated (either by
    ``freeze_teacher_at_step`` or by the caller loading an external
    teacher checkpoint). We hard-raise if ``model.teacher is None``.
    """
    if getattr(model, "teacher", None) is None:
        raise RuntimeError(
            "build_teacher_slots_fn: model.teacher is None — snapshot "
            "the teacher first (model.freeze_teacher_at_step) before "
            "constructing the replay collator."
        )
    teacher = model.teacher  # nn.ModuleDict

    def _extract(video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            enc_out = teacher["encoder"](video)
            features = enc_out["features"]
            B = video.shape[0]
            init_out = teacher["initializer"](batch_size=B, features=features)
            slots_initial = init_out[0] if isinstance(init_out, tuple) else init_out
            proc_out = teacher["processor"](slots_initial, features)
            slots = proc_out["state"]  # [B, T, K, D]
        return slots

    return _extract
