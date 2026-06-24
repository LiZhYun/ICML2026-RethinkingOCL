"""Amodal Persistence Particles (APP) — core per-frame updater.

Each particle carries (q, v, a, m):
  q ∈ R^D        — appearance prototype
  v ∈ [0,1]^P    — visible occupancy over patches
  a ∈ [0,1]^P    — amodal occupancy over patches
  m ∈ {0,1,2}    — state (0=active, 1=dormant, 2=dead)

Invariant: v(p) ≤ a(p) for every patch p.

Reference: refine-logs/FINAL_PROPOSAL_2026_04_17.md §3.1–§3.5
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from slotcontrast.utils import make_build_fn

ACTIVE = 0
DORMANT = 1
DEAD = 2


@make_build_fn(__name__, "amodal particles module")
def build(config, name: str, **kwargs):
    pass


class AmodalParticleProcessor(nn.Module):
    """Per-frame APP state updater.  Drop-in replacement for LatentProcessor
    inside ScanOverTime (same forward signature, same output keys).

    Between frames ScanOverTime passes ``state_predicted`` (= q, the appearance
    prototype) as the next ``state``.  The amodal field, particle-state flags,
    and dead-frame counter are maintained as instance-level caches that are
    reset at t=0.
    """

    def __init__(
        self,
        corrector: nn.Module,
        slot_dim: int = 64,
        feat_dim: int = 768,
        tau: float = 0.1,
        lambda_a: float = 1.0,
        alpha: float = 5.0,
        beta: float = 2.0,
        gamma: float = 3.0,
        mu: float = 0.3,
        tau_vis: float = 0.5,
        tau_alive: float = 1.0,
        tau_reid: float = 0.3,
        tau_dead: float = 0.3,
        tau_birth: float = 2.0,
        l_dead: int = 5,
        l_birth: int = 3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.corrector = corrector
        self.tau = tau
        self.lambda_a = lambda_a
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.tau_vis = tau_vis
        self.tau_alive = tau_alive
        self.tau_reid = tau_reid
        self.tau_dead = tau_dead
        self.tau_birth = tau_birth
        self.l_dead = l_dead
        self.l_birth = l_birth
        self.eps = eps

        self._prev_a: Optional[torch.Tensor] = None
        self._prev_m: Optional[torch.Tensor] = None
        self._prev_v: Optional[torch.Tensor] = None
        self._dead_counter: Optional[torch.Tensor] = None
        self._birth_accumulator: Optional[torch.Tensor] = None
        self._prev_depth: Optional[torch.Tensor] = None
        self._prev_flow: Optional[torch.Tensor] = None

    def reset(self):
        self._prev_a = None
        self._prev_m = None
        self._prev_v = None
        self._dead_counter = None
        self._birth_accumulator = None
        self._prev_depth = None
        self._prev_flow = None

    def forward(
        self,
        state: torch.Tensor,
        inputs: Optional[torch.Tensor],
        time_step: Optional[int] = None,
        init_state: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        q = state
        F_t = inputs
        B, K, D = q.shape
        P = F_t.shape[1]

        if time_step == 0 or self._prev_a is None:
            return self._first_frame(q, F_t, existence_mask, depth, flow)

        # Use *previous* frame's flow for transport: flow[:, t] is t→t+1,
        # but we need (t-1)→t to warp a_{t-1} into frame t.
        a_bar = self._transport_amodal(self._prev_a, self._prev_flow, depth, P)
        v, v_logits = self._compute_visible(q, F_t, a_bar, self._prev_m)
        o = self._compute_occlusion_plausibility(v, depth, flow, P)
        a = self._update_amodal(v, a_bar, o)
        q_new = self._update_appearance(q, v, F_t)
        prev_m_snapshot = self._prev_m.clone()
        m, existence, births_spawned = self._state_transitions(
            v, a, self._prev_m, a_bar, self._dead_counter, existence_mask
        )

        self._prev_a = a.detach()
        self._prev_m = m.detach()
        self._prev_v = v.detach()
        self._prev_depth = depth.detach() if depth is not None else None
        self._prev_flow = flow.detach() if flow is not None else None

        result = {
            "state": q_new,
            "state_predicted": q_new,
            "corrector": {"slots": q_new, "masks": v},
            "visible_masks": v,
            "amodal_masks": a,
            "amodal_transported": a_bar,
            "occlusion_plausibility": o,
            "particle_states": m,
            "prev_particle_states": prev_m_snapshot,
            "v_logits": v_logits,
        }
        if existence is not None:
            result["existence_mask"] = existence
        return result

    def _first_frame(
        self,
        q: torch.Tensor,
        F_t: torch.Tensor,
        existence_mask: Optional[torch.Tensor],
        depth: Optional[torch.Tensor],
        flow: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        B, K, D = q.shape
        P = F_t.shape[1]

        corrector_output = self.corrector(q, F_t)
        q_updated = corrector_output["slots"]
        masks = corrector_output["masks"]
        v = F.softmax(masks, dim=1)
        a = v.clone()
        m = torch.zeros(B, K, device=q.device, dtype=torch.long)
        if existence_mask is not None:
            dead = existence_mask < 0.5
            m[dead] = DEAD

        self._prev_a = a.detach()
        self._prev_m = m.detach()
        self._prev_v = v.detach()
        self._dead_counter = torch.zeros(B, K, device=q.device, dtype=torch.long)
        self._birth_accumulator = torch.zeros(B, P, device=q.device)
        self._prev_depth = depth.detach() if depth is not None else None
        self._prev_flow = flow.detach() if flow is not None else None

        # Match v_logits shape to later frames: [B, K+1, P] with bg channel
        bg_logits = torch.zeros(B, 1, P, device=q.device, dtype=q.dtype)
        v_logits = torch.cat([masks, bg_logits], dim=1)

        return {
            "state": q_updated,
            "state_predicted": q_updated,
            "corrector": {"slots": q_updated, "masks": v},
            "visible_masks": v,
            "amodal_masks": a,
            "amodal_transported": a,
            "occlusion_plausibility": torch.zeros(B, K, P, device=q.device),
            "particle_states": m,
            "prev_particle_states": m,
            "v_logits": v_logits,
        }

    def _transport_amodal(
        self,
        prev_a: torch.Tensor,
        flow: Optional[torch.Tensor],
        depth: Optional[torch.Tensor],
        P: int,
    ) -> torch.Tensor:
        """Warp previous amodal support by forward flow.

        Args:
            prev_a: [B, K, P] previous amodal occupancy.
            flow: [B, H_f, W_f, 2] forward flow (pixel units) or None.
            depth: unused here (reserved for depth-aware warp).
            P: number of patches.

        Returns:
            [B, K, P] transported amodal.
        """
        if flow is None:
            return prev_a

        B, K, _ = prev_a.shape
        H_p = W_p = int(P ** 0.5)
        if H_p * W_p != P:
            return prev_a

        a_spatial = prev_a.view(B * K, 1, H_p, W_p)
        H_f, W_f = flow.shape[1], flow.shape[2]
        if (H_f, W_f) != (H_p, W_p):
            flow_resized = F.interpolate(
                flow.permute(0, 3, 1, 2).float(),
                size=(H_p, W_p),
                mode="bilinear",
                align_corners=False,
            )
            flow_resized = flow_resized.permute(0, 2, 3, 1)
            flow_resized[..., 0] *= float(W_p) / float(W_f)
            flow_resized[..., 1] *= float(H_p) / float(H_f)
        else:
            flow_resized = flow.float()

        grid = self._flow_to_grid(H_p, W_p, flow_resized)
        grid_expanded = grid.unsqueeze(1).expand(B, K, H_p, W_p, 2).reshape(B * K, H_p, W_p, 2)

        a_warped = F.grid_sample(
            a_spatial.float(), grid_expanded,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )
        a_warped = a_warped.view(B, K, H_p * W_p).clamp(0, 1)
        return a_warped.to(prev_a.dtype)

    @staticmethod
    def _flow_to_grid(H: int, W: int, flow_hw: torch.Tensor) -> torch.Tensor:
        """Convert pixel-unit forward flow to a backward-warp sampling grid.

        Forward flow says where each source pixel goes. For backward warp
        (sampling the source for each destination), we use
        ``source = destination - flow(destination)`` as a first-order
        approximation.
        """
        B = flow_hw.shape[0]
        device, dtype = flow_hw.device, flow_hw.dtype
        ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) * (2.0 / H) - 1.0
        xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) * (2.0 / W) - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        scale = torch.tensor([2.0 / W, 2.0 / H], device=device, dtype=dtype)
        return base - flow_hw * scale

    def _compute_visible(
        self,
        q: torch.Tensor,
        F_t: torch.Tensor,
        a_bar: torch.Tensor,
        prev_m: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute visible occupancy via appearance + amodal prior.

        ℓ^v_{i}(p) = <q_i, F_t(p)> / τ  +  λ_a · log(ε + ā_i(p))
        v_{i}(p) = softmax_{i ∪ bg}(ℓ^v_{i}(p))

        Dead particles are masked to -inf so they cannot steal visible mass.

        Returns:
            v: [B, K, P] visible occupancy (softmax over K+1).
            v_logits: [B, K+1, P] pre-softmax logits (for losses).
        """
        B, K, D = q.shape
        P = F_t.shape[1]

        q_n = F.normalize(q, dim=-1)
        F_n = F.normalize(F_t, dim=-1)
        sim = torch.bmm(q_n, F_n.transpose(1, 2)) / self.tau

        amodal_prior = self.lambda_a * torch.log(self.eps + a_bar)
        logits_particles = sim + amodal_prior

        if prev_m is not None:
            dead_mask = (prev_m == DEAD).unsqueeze(-1).expand_as(logits_particles)
            logits_particles = logits_particles.masked_fill(dead_mask, float("-inf"))

        bg_logits = torch.zeros(B, 1, P, device=q.device, dtype=q.dtype)
        logits_all = torch.cat([logits_particles, bg_logits], dim=1)

        v_all = F.softmax(logits_all, dim=1)
        v = v_all[:, :K]
        return v, logits_all

    def _compute_occlusion_plausibility(
        self,
        v: torch.Tensor,
        depth: Optional[torch.Tensor],
        flow: Optional[torch.Tensor],
        P: int,
    ) -> torch.Tensor:
        """Compute per-particle per-patch occlusion plausibility.

        o_{i}(p) = clip(closer_fg_{i}(p) + fb_occ(p), 0, 1)

        closer_fg_{i}(p) is the visible mass from particles closer in depth.
        fb_occ(p) is a forward-backward flow occlusion cue (simplified: uses
        flow magnitude discontinuity as a proxy when backward flow is absent).
        """
        B, K, _ = v.shape
        device = v.device

        o = torch.zeros(B, K, P, device=device, dtype=v.dtype)

        if depth is not None:
            H_p = W_p = int(P ** 0.5)
            if H_p * W_p == P:
                depth_patches = F.adaptive_avg_pool2d(
                    depth.unsqueeze(1).float(), (H_p, W_p)
                ).view(B, 1, P)
                d_per_particle = (v * depth_patches).sum(dim=-1) / (
                    v.sum(dim=-1) + self.eps
                )
                for i in range(K):
                    closer = d_per_particle < d_per_particle[:, i : i + 1]
                    closer_fg_mass = (v * closer.unsqueeze(-1).float()).sum(dim=1)
                    o[:, i] += closer_fg_mass

        if flow is not None:
            H_p = W_p = int(P ** 0.5)
            if H_p * W_p == P:
                flow_mag = flow.float().norm(dim=-1)
                flow_patch = F.adaptive_avg_pool2d(
                    flow_mag.unsqueeze(1), (H_p, W_p)
                ).view(B, P)
                median_mag = flow_patch.median(dim=-1, keepdim=True).values
                fb_occ = (flow_patch > 2.0 * median_mag + 1.0).float()
                o = o + fb_occ.unsqueeze(1)

        return o.clamp(0, 1)

    def _update_amodal(
        self,
        v: torch.Tensor,
        a_bar: torch.Tensor,
        o: torch.Tensor,
    ) -> torch.Tensor:
        """Update amodal occupancy.

        ρ(p) = σ(α·o(p) + β·log(ε + ā(p)) - γ·(1 - v(p)))
        a(p) = 1 - (1 - v(p))·(1 - ρ(p)·ā(p))
        """
        rho = torch.sigmoid(
            self.alpha * o
            + self.beta * torch.log(self.eps + a_bar)
            - self.gamma * (1.0 - v)
        )
        a = 1.0 - (1.0 - v) * (1.0 - rho * a_bar)
        return a.clamp(0, 1)

    def _update_appearance(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        F_t: torch.Tensor,
    ) -> torch.Tensor:
        """Update appearance prototype from visible evidence only.

        q_new = normalize((1-μ)·q + μ · Σ_p v(p)·F(p) / (Σ_p v(p) + ε))
        """
        vis_weighted = torch.bmm(v, F_t)
        vis_mass = v.sum(dim=-1, keepdim=True) + self.eps
        vis_mean = vis_weighted / vis_mass

        q_new = (1.0 - self.mu) * q + self.mu * vis_mean
        return F.normalize(q_new, dim=-1)

    def _state_transitions(
        self,
        v: torch.Tensor,
        a: torch.Tensor,
        prev_m: torch.Tensor,
        a_bar: torch.Tensor,
        dead_counter: torch.Tensor,
        existence_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Compute active/dormant/dead transitions and births.

        Returns:
            m: [B, K] updated particle states.
            existence: [B, K] float existence mask (1=alive, 0=dead).
            births_spawned: [B] count of new births this frame (diagnostic).
        """
        B, K, P = v.shape
        device = v.device

        vis_mass = v.sum(dim=-1)
        support = a.sum(dim=-1)

        m = prev_m.clone()

        # Reactivation check FIRST on particles that were dormant BEFORE
        # this step (avoids race where active→dormant→active in one step).
        was_dormant = prev_m == DORMANT
        if was_dormant.any():
            reapp_score = (v * a_bar).sum(dim=-1)
            reactivate = was_dormant & (reapp_score >= self.tau_reid)
            m[reactivate] = ACTIVE

        # Then check active→dormant on particles still active after reactivation.
        was_active = prev_m == ACTIVE
        low_vis = vis_mass < self.tau_vis
        high_support = support >= self.tau_alive
        become_dormant = was_active & low_vis & high_support
        m[become_dormant] = DORMANT

        # Death: low amodal support for L_dead consecutive frames.
        low_support = support < self.tau_dead
        alive = m != DEAD
        dc = dead_counter.clone()
        dc[alive & low_support] += 1
        dc[alive & ~low_support] = 0
        should_die = alive & (dc >= self.l_dead)
        m[should_die] = DEAD
        self._dead_counter = dc

        births_spawned = torch.zeros(B, device=device, dtype=torch.long)

        existence = (m != DEAD).float()
        return m, existence, births_spawned
