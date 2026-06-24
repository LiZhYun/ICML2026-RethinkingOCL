from typing import Any, Dict, List, Mapping, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from slotcontrast.utils import make_build_fn


@make_build_fn(__name__, "video module")
def build(config, name: str, **kwargs):
    pass  # No special module building needed


class LatentProcessor(nn.Module):
    """Updates latent state based on inputs and state and predicts next state."""

    def __init__(
        self,
        corrector: nn.Module,
        predictor: Optional[nn.Module] = None,
        memory_encoder: Optional[nn.Module] = None,
        memory_bank: Optional[nn.Module] = None,
        state_key: str = "slots",
        first_step_corrector_args: Optional[Dict[str, Any]] = None,
        use_ttt3r: bool = False,
        use_cycle_consistency: bool = False,
        skip_corrector: bool = False,
        skip_predictor: bool = False,
        slot_drop_prob: float = 0.0,
        drop_fraction: float = 0.3,
    ):
        super().__init__()
        self.corrector = corrector
        self.predictor = predictor
        self.memory_encoder = memory_encoder
        self.memory_bank = memory_bank
        self.state_key = state_key
        self.use_ttt3r = use_ttt3r
        self.use_cycle_consistency = use_cycle_consistency
        self.skip_corrector = skip_corrector
        self.skip_predictor = skip_predictor
        if first_step_corrector_args is not None:
            self.first_step_corrector_args = first_step_corrector_args
        else:
            self.first_step_corrector_args = None

        # Slot-Drop-Recover (Idea #012). Per-frame Bernoulli gate with probability
        # ``slot_drop_prob``; when the gate fires, zero a random ``drop_fraction``
        # of slot positions in ``prev_slots`` before they reach the corrector. A
        # pre-drop snapshot is exported so ``SlotDropRecoverLoss`` can pull dropped
        # identities back toward their original features via the corrector's
        # current-frame evidence.
        if not (0.0 <= slot_drop_prob <= 1.0):
            raise ValueError(
                f"`slot_drop_prob` must be in [0, 1], got {slot_drop_prob}."
            )
        if not (0.0 <= drop_fraction <= 1.0):
            raise ValueError(
                f"`drop_fraction` must be in [0, 1], got {drop_fraction}."
            )
        self.slot_drop_prob = float(slot_drop_prob)
        self.drop_fraction = float(drop_fraction)

        # Cache previous masks for IoU cost in HungarianPredictor
        self._prev_masks: Optional[torch.Tensor] = None

    def forward(
        self, state: torch.Tensor, inputs: Optional[torch.Tensor], time_step: Optional[int] = None,
        init_state: Optional[torch.Tensor] = None, existence_mask: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # state: batch x n_slots x slot_dim
        assert state.ndim == 3
        # inputs: batch x n_patches x input_dim (encoder features)
        assert inputs.ndim == 3

        # 0. SLOT-DROP-RECOVER (Idea #012). Before the corrector sees the
        # previous-frame slots, optionally snapshot them and zero a random subset
        # so the corrector must recover the dropped identities from the current
        # frame's features. Only active during training, after t=0 (where
        # "previous slots" are meaningful), and when the per-frame Bernoulli gate
        # fires. Exports ``pre_drop_slots`` (detached) and ``drop_mask`` on the
        # per-frame output dict; ``ScanOverTime.merge_dict_trees`` stacks these
        # along the time axis to yield ``[B, T, N, D]`` / ``[B, T, N]`` tensors.
        # Reproducibility: relies on the global PyTorch seed set by the trainer
        # (no private generator is maintained).
        pre_drop_slots = state.detach().clone()  # [B, N, D]
        initial_queries_snapshot = state  # captured BEFORE the drop for cycle-consistency parity
        B, N, _ = state.shape
        drop_mask = torch.zeros(B, N, dtype=torch.bool, device=state.device)
        if (
            self.training
            and self.slot_drop_prob > 0.0
            and time_step is not None
            and time_step > 0
        ):
            # Per-frame Bernoulli gate: each frame independently decides whether
            # to drop. A per-video coin would be too coarse (either all frames or
            # none); a per-slot gate alone would not control fraction. We use
            # per-frame gate * fixed fraction.
            gate = torch.rand(B, device=state.device) < self.slot_drop_prob  # [B]
            if gate.any():
                # Sample ``drop_fraction * N`` slot positions uniformly at random
                # for each gated batch element via top-k on random noise. This
                # guarantees a fixed drop count per sampled video.
                n_drop = int(round(self.drop_fraction * N))
                if n_drop > 0:
                    noise = torch.rand(B, N, device=state.device)
                    # Restrict drops to existing slot positions when an
                    # existence_mask is provided: push noise for non-existing
                    # slots to -inf so they are never selected.
                    if existence_mask is not None:
                        em = existence_mask
                        if em.ndim == 3:  # [B, T, N] -> [B, N] at this frame
                            em = em[:, time_step]
                        em_f = em.to(device=state.device, dtype=noise.dtype)
                        noise = noise * em_f + (em_f - 1.0) * 1e9
                    _, drop_idx = noise.topk(n_drop, dim=-1)  # [B, n_drop]
                    frame_mask = torch.zeros_like(drop_mask)
                    frame_mask.scatter_(1, drop_idx, True)
                    # Only apply drops to gated batch elements.
                    frame_mask = frame_mask & gate.unsqueeze(-1)
                    drop_mask = frame_mask
                    # Zero the selected slot positions in the state passed to
                    # the corrector (broadcast mask to the feature dim).
                    state = state * (~drop_mask).unsqueeze(-1).to(state.dtype)

        # 1. CORRECT: Update slots based on current frame features
        if self.skip_corrector:
            # Bypass slot attention - use input state directly
            corrector_output = {"slots": state}
            updated_state = state
            curr_masks = None
        elif inputs is not None:
            if time_step == 0 and self.first_step_corrector_args:
                corrector_output = self.corrector(state, inputs, **self.first_step_corrector_args)
            else:
                corrector_output = self.corrector(state, inputs)
            updated_state = corrector_output[self.state_key]
            curr_masks = corrector_output.get("masks")
        else:
            # Run predictor without updating on current inputs
            corrector_output = None
            updated_state = state
            curr_masks = None

        # 2. ENCODE MEMORY (if components exist)
        if (
            self.memory_encoder is not None
            and self.memory_bank is not None
            and time_step is not None
            and curr_masks is not None
        ):
            # Encode current frame into memory
            memory_encoding = self.memory_encoder(
                slots=updated_state.detach(),
                features=inputs.detach(),
                masks=curr_masks.detach(),
            )

            # Store in memory bank
            self.memory_bank.push(
                frame_idx=time_step,
                slots=updated_state.detach(),
                features=inputs.detach(),
                masks=curr_masks.detach(),
                **memory_encoding
            )

        # 3. RETRIEVE MEMORY for prediction
        memory = None
        memory_pos = None
        if self.memory_bank is not None and time_step is not None and time_step > 0:
            memory, memory_pos = self.memory_bank.get_memories(time_step, training=self.training)

        # 4. PREDICT: Generate initialization for NEXT frame
        attn_list = None
        out_existence_mask = existence_mask  # Default: pass through
        
        # Compute hybrid cost inputs if predictor uses it
        curr_centroids = None
        if self.predictor and hasattr(self.predictor, 'use_hybrid_cost') and self.predictor.use_hybrid_cost:
            # Compute centroids from corrector masks
            if curr_masks is not None:
                from slotcontrast.modules.initializers import compute_slot_centroids_from_masks
                patch_h = int(inputs.shape[1] ** 0.5)
                curr_centroids = compute_slot_centroids_from_masks(
                    curr_masks, patch_h, patch_h, normalize=True
                )
        
        if self.predictor and not self.skip_predictor:
            use_memory = (
                hasattr(self.predictor, "use_memory")
                and getattr(self.predictor, "use_memory", False)
                and memory is not None
            )
            # Check if predictor supports init_state (CrossAttentionPredictor)
            use_init_state = hasattr(self.predictor, 'cross_attn') and init_state is not None
            # Check if predictor is HungarianPredictor (matching-based)
            is_hungarian = hasattr(self.predictor, '_hungarian_match')
            # Check if predictor is HungarianMemoryMatcher (has max_slots for persistent registry)
            is_memory_matcher = hasattr(self.predictor, 'max_slots')
            
            if use_memory:
                result = self.predictor(
                    updated_state, memory, memory_pos, return_weights=self.use_ttt3r
                )
            elif is_memory_matcher and existence_mask is not None:
                # HungarianMemoryMatcher with variable slots
                result = self.predictor(updated_state, existence_mask=existence_mask, return_weights=True)
                predicted_state, out_existence_mask = result[0], result[1]
            elif use_init_state:
                result = self.predictor(
                    updated_state, init_state=init_state, return_weights=self.use_ttt3r
                )
            elif is_hungarian and existence_mask is not None:
                # HungarianPredictor with existence_mask: reorder both slots and mask
                result = self.predictor(
                    updated_state, existence_mask=existence_mask, return_weights=True,
                    centroids=curr_centroids, masks=curr_masks, prev_masks=self._prev_masks,
                    features=inputs, flow=flow, depth=depth, slot_masks=curr_masks,
                )
                predicted_state, out_existence_mask = result[0], result[1]
            elif is_hungarian:
                # HungarianPredictor uses internal state, just pass slots
                result = self.predictor(
                    updated_state, return_weights=self.use_ttt3r,
                    centroids=curr_centroids, masks=curr_masks, prev_masks=self._prev_masks,
                    features=inputs, flow=flow, depth=depth, slot_masks=curr_masks,
                )
            else:
                result = self.predictor(updated_state, return_weights=self.use_ttt3r)
            
            # Parse result based on return_weights (skip if already handled by memory matcher)
            if not (is_memory_matcher and existence_mask is not None):
                if self.use_ttt3r and isinstance(result, tuple):
                    predicted_state, attn_list = result[0], result[-1]
                else:
                    predicted_state = result if not isinstance(result, tuple) else result[0]
            
            # TTT3R: Adaptive update based on attention
            if self.use_ttt3r and time_step is not None and time_step > 0 and attn_list is not None:
                predicted_state = self._apply_ttt3r_update(
                    updated_state, predicted_state, attn_list
                )
        else:
            predicted_state = updated_state
        
        # Cache current masks for next frame's IoU cost
        if curr_masks is not None:
            self._prev_masks = curr_masks.detach()

        # Determine output state: use reordered slots if Hungarian post-match
        # (Hungarian post-match reorders slots to match temporal identity, so decoder should use reordered)
        is_hungarian_postmatch = (
            self.predictor is not None
            and hasattr(self.predictor, '_hungarian_match')
            and not getattr(self.predictor, 'pre_match', False)
        )
        output_state = predicted_state if is_hungarian_postmatch else updated_state

        result = {
            "state": output_state,
            "state_predicted": predicted_state,
            "corrector": corrector_output,
            "initial_queries": initial_queries_snapshot,  # Store for cycle consistency (pre-drop)
            # Slot-Drop-Recover (Idea #012): emit per-frame snapshot and mask so
            # that after ScanOverTime stacking they become [B, T, N, D] and
            # [B, T, N] tensors consumable by SlotDropRecoverLoss.
            "pre_drop_slots": pre_drop_slots,
            "drop_mask": drop_mask,
        }

        # Add existence mask if available (from memory matcher)
        if out_existence_mask is not None:
            result["existence_mask"] = out_existence_mask

        # Predictor analysis: measure if predictor is ~identity
        if self.predictor and not self.skip_predictor:
            with torch.no_grad():
                # Cosine similarity (1.0 = identical direction)
                cos_sim = F.cosine_similarity(updated_state, predicted_state, dim=-1).mean()
                # Relative change (0.0 = identical)
                delta_norm = (predicted_state - updated_state).norm()
                input_norm = updated_state.norm() + 1e-8
                rel_change = delta_norm / input_norm
                result["predictor_cos_sim"] = cos_sim
                result["predictor_rel_change"] = rel_change

        return result

    def _apply_ttt3r_update(
        self,
        current_state: torch.Tensor,
        predicted_state: torch.Tensor,
        attn_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Apply TTT3R adaptive update based on attention weights.
        
        Following TTT3R: use mean attention as relevance score.
        Higher attention = more relevant = larger update weight.
        """
        # attn_list: list of [B, n_slots, n_keys] from each layer
        if len(attn_list) == 0 or all(a is None for a in attn_list):
            return predicted_state
        
        # Stack and rearrange: [n_layers, B, n_slots, n_keys] -> [B, n_slots, n_keys, n_layers]
        attn_stack = torch.stack([a for a in attn_list if a is not None], dim=0)
        attn_stack = attn_stack.permute(1, 2, 3, 0)  # [B, n_slots, n_keys, n_layers]
        
        # Mean over keys and layers: [B, n_slots]
        state_relevance = attn_stack.mean(dim=(-1, -2))
        
        # Sigmoid scaling and broadcast: [B, n_slots, 1]
        update_weight = torch.sigmoid(state_relevance).unsqueeze(-1)
        
        # Adaptive blend: predicted_state * weight + current_state * (1 - weight)
        return predicted_state * update_weight + current_state * (1 - update_weight)


class MapOverTime(nn.Module):
    """Wrapper applying wrapped module independently to each time step.

    Assumes batch is first dimension, time is second dimension.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        batch_size = None
        seq_len = None
        flattened_args = []
        for idx, arg in enumerate(args):
            if arg is None:
                flattened_args.append(None)
                continue
            B, T = arg.shape[:2]
            if not batch_size:
                batch_size = B
            elif batch_size != B:
                raise ValueError(
                    f"Inconsistent batch size of {B} of argument {idx}, was {batch_size} before."
                )

            if not seq_len:
                seq_len = T
            elif seq_len != T:
                raise ValueError(
                    f"Inconsistent sequence length of {T} of argument {idx}, was {seq_len} before."
                )

            flattened_args.append(arg.flatten(0, 1))

        # Handle camera_data dict in kwargs (flatten each tensor inside)
        flattened_kwargs = {}
        for key, val in kwargs.items():
            if val is None:
                flattened_kwargs[key] = None
            elif isinstance(val, dict):
                flattened_kwargs[key] = {
                    k: v.flatten(0, 1) if isinstance(v, torch.Tensor) else v
                    for k, v in val.items()
                }
            elif isinstance(val, torch.Tensor):
                flattened_kwargs[key] = val.flatten(0, 1)
            else:
                flattened_kwargs[key] = val

        outputs = self.module(*flattened_args, **flattened_kwargs)

        if isinstance(outputs, Mapping):
            unflattened_outputs = {
                k: v.unflatten(0, (batch_size, seq_len)) for k, v in outputs.items()
            }
        else:
            unflattened_outputs = outputs.unflatten(0, (batch_size, seq_len))

        return unflattened_outputs


class ScanOverTime(nn.Module):
    """Wrapper applying wrapped module recurrently over time steps"""

    def __init__(
        self, module: nn.Module, next_state_key: str = "state_predicted", pass_step: bool = True
    ) -> None:
        super().__init__()
        self.module = module
        self.next_state_key = next_state_key
        self.pass_step = pass_step

    def forward(self, initial_state: torch.Tensor, inputs: torch.Tensor,
                existence_mask: Optional[torch.Tensor] = None,
                flow: Optional[torch.Tensor] = None,
                depth: Optional[torch.Tensor] = None):
        # initial_state: [B, n_slots, D] or [B, T, n_slots, D] for per-frame init
        # inputs: batch x n_frames x ...
        # existence_mask: [B, n_slots] or [B, T, n_slots] for variable slots
        # flow: optional [B, T-1, H, W, 2] forward optical flow (frame t -> t+1).
        #       Some sources deliver [B, T, H, W, 2] padded; either is accepted.
        # depth: optional [B, T, H, W] VGGT depth map; sliced per-step to [B, H, W].
        seq_len = inputs.shape[1]
        per_frame_init = initial_state.ndim == 4  # [B, T, n_slots, D]
        per_frame_mask = existence_mask is not None and existence_mask.ndim == 3  # [B, T, n_slots]
        flow_len = flow.shape[1] if flow is not None else 0
        cost_margin_list: Optional[List[Optional[torch.Tensor]]] = None

        # Clear memory bank at start of sequence
        if hasattr(self.module, "memory_bank") and self.module.memory_bank is not None:
            self.module.memory_bank.clear()
        
        # Reset APP particle processor state at start of sequence
        if hasattr(self.module, "reset") and callable(self.module.reset):
            self.module.reset()

        # Reset predictor state at start of sequence
        is_hungarian = (
            hasattr(self.module, "predictor")
            and hasattr(self.module.predictor, "_hungarian_match")
        )
        is_memory_matcher = (
            hasattr(self.module, "predictor")
            and hasattr(self.module.predictor, "max_slots")
        )
        if hasattr(self.module, "predictor") and hasattr(self.module.predictor, "reset"):
            self.module.predictor.reset()

        # H1 Cutie-Slot memory bank: allocate per-clip ragged buffers sized to
        # the current batch. CutieSlotPredictor.reset() cannot do this alone
        # because it has no batch-size info; we pass it explicitly here. The
        # ``hasattr`` guard keeps the standard predictors untouched.
        if (
            hasattr(self.module, "predictor")
            and hasattr(self.module.predictor, "memory_bank")
            and self.module.predictor.memory_bank is not None
            and hasattr(self.module.predictor.memory_bank, "reset")
        ):
            self.module.predictor.memory_bank.reset(inputs.shape[0])

        # Check if pre-matching mode is enabled (True or "greedy")
        use_pre_match = (
            is_hungarian
            and not is_memory_matcher  # Memory matcher handles alignment internally
            and hasattr(self.module.predictor, "pre_match")
            and self.module.predictor.pre_match
        )

        state = initial_state[:, 0] if per_frame_init else initial_state
        outputs = []
        match_indices_list = [] if is_hungarian else None
        if is_hungarian:
            cost_margin_list = []
        
        for t in range(seq_len):
            # Get existence mask for this frame
            mask_t = existence_mask[:, t] if per_frame_mask else existence_mask

            # Slice flow for this step: flow[:, t] is the t -> t+1 motion.
            # For the last frame (t == seq_len - 1) there is no next frame, so
            # flow_t must be None regardless of how the flow tensor was padded.
            # Callers may pass [B, T-1, ...] (flow_len == T-1) or
            # [B, T, ...] (flow_len == T, with a dummy trailing entry). The
            # ``seq_len - 1`` bound defends against the padded variant whose
            # last slot is bogus (Codex-review MAJOR #5).
            if flow is not None and t < min(flow_len, seq_len - 1):
                flow_t = flow[:, t]
            else:
                flow_t = None

            # For per-frame init with Hungarian (not memory matcher)
            if per_frame_init and t > 0:
                if is_memory_matcher:
                    # Memory matcher handles alignment internally
                    state = initial_state[:, t]
                elif use_pre_match:
                    # Pre-match: align greedy init to reference BEFORE slot attention
                    state = self.module.predictor.match_to_reference(initial_state[:, t])
                else:
                    # Post-match (original): use fresh init, Hungarian matches after slot attention
                    state = initial_state[:, t]

            init_state_t = initial_state[:, t] if per_frame_init else None
            # Slice depth for this step: depth[:, t] is the t-th frame's [B, H, W]
            # map. We do not pad; missing depth is a config error, not a runtime
            # fallback (hard-raise policy).
            depth_t = depth[:, t] if depth is not None else None
            # Only forward `flow=` when the caller actually supplied flow, so
            # modules whose `forward` signature does not declare a `flow` kwarg
            # (e.g. the dummy RecurrentCell in tests) keep working.
            extra_kwargs = {"flow": flow_t} if flow is not None else {}
            if depth is not None:
                extra_kwargs["depth"] = depth_t
            if self.pass_step:
                output = self.module(
                    state, inputs[:, t], t,
                    init_state=init_state_t, existence_mask=mask_t, **extra_kwargs,
                )
            else:
                output = self.module(
                    state, inputs[:, t],
                    init_state=init_state_t, existence_mask=mask_t, **extra_kwargs,
                )
            outputs.append(output)
            state = output[self.next_state_key]
            
            # Collect Hungarian match indices
            if is_hungarian:
                indices = self.module.predictor.get_last_match_indices()
                match_indices_list.append(indices)
                cost_margin_list.append(getattr(self.module.predictor, "_last_cost_margin", None))

        result = merge_dict_trees(outputs, axis=1)

        # Add match indices to result [B, T, N] or None for first frame
        if match_indices_list is not None:
            result["hungarian_match_indices"] = match_indices_list
        if cost_margin_list is not None:
            result["hungarian_cost_margin"] = cost_margin_list

        return result


def merge_dict_trees(trees: List[Mapping], axis: int = 0):
    """Stack all leafs given a list of dictionaries trees.

    Example:
    x = merge_dict_trees([
        {
            "a": torch.ones(2, 1),
            "b": {"x": torch.ones(2, 2)}
        },
        {
            "a": torch.ones(3, 1),
            "b": {"x": torch.ones(1, 2)}
        }
    ])

    x == {
        "a": torch.ones(5, 1),
        "b": {"x": torch.ones(3, 2)}
    }
    """
    out = {}
    if len(trees) > 0:
        ref_tree = trees[0]
        for key, value in ref_tree.items():
            values = [tree[key] for tree in trees]
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    # Handle scalar tensors - stack along new dimension (axis 0)
                    out[key] = torch.stack(values, 0)
                else:
                    out[key] = torch.stack(values, axis)
            elif isinstance(value, Mapping):
                out[key] = merge_dict_trees(values, axis)
            else:
                out[key] = values

    return out
