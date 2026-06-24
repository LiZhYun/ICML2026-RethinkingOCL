from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torchvision.utils import make_grid

from slotcontrast import configuration, losses, modules, optimizers, utils, visualizations
from slotcontrast.data.transforms import Denormalize


def build(
    model_config: configuration.ModelConfig,
    optimizer_config,
    train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
):
    optimizer_builder = optimizers.OptimizerBuilder(**optimizer_config)

    initializer = modules.build_initializer(model_config.initializer)
    # Snapshot RNG state before encoder construction so LoRA insertion (which
    # consumes RNG via lora_A.kaiming_uniform / lora_B.zeros init) does not
    # shift the random init of downstream modules. Round 13 reviewer concern:
    # frozen vs rescue runs must have identical grouper/decoder/predictor
    # initialisations to isolate the rescue protocol's effect.
    _rng_state = torch.get_rng_state()
    encoder = modules.build_encoder(
        model_config.encoder, default_name="FrameEncoder"
    )
    torch.set_rng_state(_rng_state)
    grouper = modules.build_grouper(model_config.grouper)
    decoder = modules.build_decoder(model_config.decoder)

    target_encoder = None
    if model_config.get("target_encoder"):
        # Snapshot/restore RNG around target_encoder construction so the
        # video predictor (built later) sees identical RNG in frozen-vs-rescue
        # paired comparisons. Without this snapshot, building the additional
        # frozen DINOv2 instance shifts RNG and changes predictor init.
        _tgt_rng_state = torch.get_rng_state()
        target_encoder = modules.build_encoder(
            model_config.target_encoder, default_name="FrameEncoder"
        )
        torch.set_rng_state(_tgt_rng_state)
        assert (
            model_config.get("target_encoder_input") is not None
        ), "Please specify `target_encoder_input`."

    dynamics_predictor = None
    if model_config.get("dynamics_predictor"):
        dynamics_predictor = modules.build_dynamics_predictor(model_config.dynamics_predictor)

    input_type = model_config.get("input_type", "image")
    if input_type == "image":
        processor = modules.LatentProcessor(grouper, predictor=None)
    elif input_type == "video":
        encoder = modules.MapOverTime(encoder)
        decoder = modules.MapOverTime(decoder)
        if target_encoder:
            target_encoder = modules.MapOverTime(target_encoder)
        if model_config.predictor is not None:
            predictor = modules.build_module(model_config.predictor)
        else:
            predictor = None
        
        # Build memory components if specified
        memory_encoder = None
        memory_bank = None
        if model_config.latent_processor:
            latent_proc_config = model_config.latent_processor
            proc_type = latent_proc_config.get("processor_type", "latent")

            if proc_type == "amodal_particles":
                app_kwargs = {
                    k: v for k, v in latent_proc_config.items()
                    if k not in ("processor_type", "memory_encoder", "memory_bank",
                                 "first_step_corrector_args")
                }
                processor = modules.AmodalParticleProcessor(
                    corrector=grouper, **app_kwargs
                )
            else:
                if hasattr(latent_proc_config, "memory_encoder") and latent_proc_config.memory_encoder:
                    memory_encoder = modules.build_memory_encoder(latent_proc_config.memory_encoder)
                if hasattr(latent_proc_config, "memory_bank") and latent_proc_config.memory_bank:
                    memory_bank = modules.build_memory_bank(latent_proc_config.memory_bank)

                filtered_config = {k: v for k, v in latent_proc_config.items()
                                 if k not in ("memory_encoder", "memory_bank", "processor_type")}

                processor = modules.build_video(
                    filtered_config,
                    "LatentProcessor",
                    corrector=grouper,
                    predictor=predictor,
                    memory_encoder=memory_encoder,
                    memory_bank=memory_bank,
                )
        else:
            processor = modules.LatentProcessor(grouper, predictor)
        processor = modules.ScanOverTime(processor)
    else:
        raise ValueError(f"Unknown input type {input_type}")

    target_type = model_config.get("target_type", "features")
    if target_type == "input":
        default_target_key = input_type
    elif target_type == "features":
        if model_config.target_encoder_input is not None:
            default_target_key = "target_encoder.backbone_features"
        else:
            default_target_key = "encoder.backbone_features"
    else:
        raise ValueError(f"Unknown target type {target_type}. Should be `input` or `features`.")

    loss_defaults = {
        "pred_key": "decoder.reconstruction",
        "target_key": default_target_key,
        "video_inputs": input_type == "video",
        "patch_inputs": target_type == "features",
    }
    if model_config.losses is None:
        loss_fns = {"mse": losses.build(dict(**loss_defaults, name="MSELoss"))}
    else:
        loss_fns = {
            name: losses.build({**loss_defaults, **loss_config})
            for name, loss_config in model_config.losses.items()
        }

    if model_config.mask_resizers:
        mask_resizers = {
            name: modules.build_utils(resizer_config, "Resizer")
            for name, resizer_config in model_config.mask_resizers.items()
        }
    else:
        mask_resizers = {
            "decoder": modules.build_utils(
                {
                    "name": "Resizer",
                    # When using features as targets, assume patch-shaped outputs. With other
                    # targets, assume spatial outputs.
                    "patch_inputs": target_type == "features",
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
            "grouping": modules.build_utils(
                {
                    "name": "Resizer",
                    "patch_inputs": True,
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
        }

    if model_config.masks_to_visualize:
        masks_to_visualize = model_config.masks_to_visualize
    else:
        masks_to_visualize = "decoder"

    # Check if cycle consistency loss is enabled
    use_cycle_consistency = (
        model_config.losses is not None
        and "loss_cycle" in model_config.losses
        # and model_config.get("loss_weights", {}).get("loss_cycle", 0.0) != 0.0
    )
    # Window for temporal cross-consistency (0 = same-frame only)
    temporal_cross_window = model_config.get("temporal_cross_window", 0)
    temporal_cross_mode = model_config.get("temporal_cross_mode", "both")

    # GSRS open-set identity head (proposal §3.1(a), §3.13). Optional; only
    # built when the config declares an `open_set_head` block.
    open_set_head = None
    if model_config.get("open_set_head") is not None:
        open_set_head = modules.build_module(
            model_config.open_set_head, default_group="networks"
        )

    # GSRS teacher-snapshot step (§3.12). 0 = never snapshot.
    teacher_snapshot_step = int(model_config.get("teacher_snapshot_step", 0) or 0)

    # GSRS frozen-renderer checkpoint (Part A writes this). Loaded lazily
    # when the replay branch runs; its path is stashed on the model as a
    # plain attribute. We hard-raise on a non-empty but missing path — no
    # silent fallback, per the project-wide no-fallback rule.
    gsrs_renderer_ckpt = model_config.get("gsrs_renderer_ckpt", None)
    if gsrs_renderer_ckpt is not None and gsrs_renderer_ckpt != "":
        import os as _os
        if not _os.path.exists(gsrs_renderer_ckpt):
            raise FileNotFoundError(
                f"GSRS renderer checkpoint not found at {gsrs_renderer_ckpt!r}. "
                "Set `model.gsrs_renderer_ckpt` to the path produced by the "
                "Part A renderer-pretrain script, or remove it from the YAML."
            )

    ema_config = model_config.get("ema_teacher", None)

    model = ObjectCentricModel(
        optimizer_builder,
        initializer,
        encoder,
        processor,
        decoder,
        loss_fns,
        loss_weights=model_config.get("loss_weights", None),
        target_encoder=target_encoder,
        dynamics_predictor=dynamics_predictor,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        mask_resizers=mask_resizers,
        input_type=input_type,
        target_encoder_input=model_config.get("target_encoder_input", None),
        visualize=model_config.get("visualize", False),
        visualize_every_n_steps=model_config.get("visualize_every_n_steps", 1000),
        masks_to_visualize=masks_to_visualize,
        use_cycle_consistency=use_cycle_consistency,
        temporal_cross_window=temporal_cross_window,
        temporal_cross_mode=temporal_cross_mode,
        use_backbone_features=model_config.get("use_backbone_features", False),
        initializer_feature_source=model_config.get("initializer_feature_source", "encoder"),
        use_projected_anchor=model_config.get("use_projected_anchor", False),
        open_set_head=open_set_head,
        teacher_snapshot_step=teacher_snapshot_step,
        gsrs_renderer_ckpt=gsrs_renderer_ckpt,
        ema_config=ema_config,
    )

    if model_config.load_weights:
        model.load_weights_from_checkpoint(model_config.load_weights, model_config.modules_to_load)

    return model


class ObjectCentricModel(pl.LightningModule):
    def __init__(
        self,
        optimizer_builder: Callable,
        initializer: nn.Module,
        encoder: nn.Module,
        processor: nn.Module,
        decoder: nn.Module,
        loss_fns: Dict[str, losses.Loss],
        *,
        loss_weights: Optional[Dict[str, float]] = None,
        target_encoder: Optional[nn.Module] = None,
        dynamics_predictor: Optional[nn.Module] = None,
        train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        mask_resizers: Optional[Dict[str, modules.Resizer]] = None,
        input_type: str = "image",
        target_encoder_input: Optional[str] = None,
        visualize: bool = False,
        visualize_every_n_steps: Optional[int] = None,
        masks_to_visualize: Union[str, List[str]] = "decoder",
        use_cycle_consistency: bool = False,
        temporal_cross_window: int = 0,
        temporal_cross_mode: str = "both",
        use_backbone_features: bool = False,
        initializer_feature_source: str = "encoder",
        use_projected_anchor: bool = False,
        open_set_head: Optional[nn.Module] = None,
        teacher_snapshot_step: int = 0,
        gsrs_renderer_ckpt: Optional[str] = None,
        ema_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.optimizer_builder = optimizer_builder
        self.initializer = initializer
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.target_encoder = target_encoder
        self.dynamics_predictor = dynamics_predictor
        self.use_cycle_consistency = use_cycle_consistency
        self.temporal_cross_window = temporal_cross_window
        self.temporal_cross_mode = temporal_cross_mode
        self.use_backbone_features = use_backbone_features
        # Round-24 evolution: when set to "target_encoder", greedy/saliency init
        # consumes the frozen pretrained target_encoder features instead of the
        # online (LoRA-finetuned) student features. This stabilizes per_frame
        # slot identity under feature finetuning. Default "encoder" preserves
        # legacy behaviour.
        self.initializer_feature_source = initializer_feature_source
        # Round-24 Fix #3: project-anchor mechanism. Snapshot
        # ``encoder.module.output_transform`` at the LoRA unfreeze step and
        # apply it to ``target_encoder.backbone_features`` to obtain a stable
        # 128d projected-feature target. The companion loss anchors student
        # ``encoder.features`` against this target so LoRA cannot destroy the
        # post-output_transform manifold that downstream slot attention relies
        # on. Active only when ``model.use_projected_anchor: true``.
        self.use_projected_anchor: bool = bool(use_projected_anchor)
        self._snapshot_output_transform: Optional[nn.Module] = None
        if initializer_feature_source not in ("encoder", "target_encoder"):
            raise ValueError(
                f"initializer_feature_source must be 'encoder' or 'target_encoder', "
                f"got {initializer_feature_source!r}"
            )
        if initializer_feature_source == "target_encoder" and target_encoder is None:
            raise ValueError(
                "initializer_feature_source='target_encoder' requires "
                "model.target_encoder to be configured."
            )
        # GSRS (§3.1(a), §3.12, §3.13) — all opt-in via explicit attrs.
        # Registered as a submodule so its params join the optimizer graph
        # and state-dict; when ``None`` the model behaves exactly as the
        # upstream GCv1 path (no path through `self.open_set_head` is
        # activated unless it is a real Module).
        self.open_set_head = open_set_head
        self.teacher_snapshot_step = int(teacher_snapshot_step)
        self.gsrs_renderer_ckpt = gsrs_renderer_ckpt
        # Frozen teacher is populated lazily by ``freeze_teacher_at_step``;
        # we do NOT register it as a submodule until that call fires, so
        # its parameters do not pollute the optimizer or the state dict
        # during the Phase 1 warmup (§3.12).
        self.teacher: Optional[nn.Module] = None
        # Idempotence guard for ``freeze_teacher_at_step``; re-firing the
        # call without bumping the step is a silent no-op.
        self._teacher_snapshot_done: bool = False

        # TubeGram teacher — snapshot-at-unfreeze (Round 2 A1 fix).
        # The unfreeze step is the SINGLE control knob: at this step we
        # (1) snapshot the full student pipeline (encoder, initializer,
        # processor, decoder) into a FROZEN teacher, (2) flip LoRA params to
        # trainable, and (3) begin ramping TubeGram/ChiBoost weights.
        # Snapshot (not EMA) because: (a) captures Phase-1-end state where
        # masks are already reliable (GCv1 frozen-backbone ≈ 0.62 FG-ARI on
        # MOVi-D), (b) LoRA weights are ~0 at unfreeze so snapshot = pure
        # DINOv2, (c) anchors student to a fixed reference — no moving target.
        # This addresses Round 1 Critical #1/#2/#3 (teacher-defined tubes +
        # no chicken-and-egg) and Round 2 Codex's recommendation of
        # Option (ii) snapshot-at-unfreeze over full EMA pipeline.
        self._ema_encoder: Optional[nn.Module] = None
        self._ema_initializer: Optional[nn.Module] = None
        self._ema_processor: Optional[nn.Module] = None
        self._ema_decoder: Optional[nn.Module] = None
        self._ema_decay: float = 0.0
        self._backbone_unfreeze_step: int = 0
        self._loss_ramp_steps: int = 0
        self._backbone_unfrozen: bool = False
        # Round 2 Codex Phase B fix (#2): which losses participate in the
        # phase-schedule weight ramp is now driven by config so downstream
        # cells (e.g., ``loss_globalgram``) ramp identically to tubegram/
        # chiboost. Default keeps the TubeGram proposal's pair for
        # backward-compatible behaviour.
        self._ramped_loss_names: Tuple[str, ...] = ("loss_tubegram", "loss_chiboost")
        if ema_config is not None:
            if "warmup_steps" in ema_config:
                raise ValueError(
                    "`ema_teacher.warmup_steps` was removed in the Round-1 "
                    "phase-schedule refactor. Use `backbone_unfreeze_step` "
                    "(same semantics: step at which to unfreeze LoRA + "
                    "snapshot EMA) and `loss_ramp_steps` (how many steps to "
                    "ramp TubeGram/ChiBoost weights from 0 after unfreeze)."
                )
            self._ema_decay = float(ema_config.get("decay", 0.999))
            self._backbone_unfreeze_step = int(
                ema_config.get("backbone_unfreeze_step", 0)
            )
            self._loss_ramp_steps = int(ema_config.get("loss_ramp_steps", 0))
            ramp_names = ema_config.get("ramp_loss_names", None)
            if ramp_names is not None:
                # Accept list/tuple and OmegaConf ListConfig transparently —
                # the latter is what YAML lists deserialise to.
                try:
                    ramp_names_iter = list(ramp_names)
                except TypeError as exc:
                    raise TypeError(
                        "`ema_teacher.ramp_loss_names` must be an iterable "
                        f"of loss keys, got {type(ramp_names).__name__}."
                    ) from exc
                self._ramped_loss_names = tuple(str(x) for x in ramp_names_iter)
            # Phase 1 (before `_backbone_unfreeze_step`): freeze LoRA adapters
            # so the backbone cannot drift while slot attention + decoder
            # stabilise. This runs INDEPENDENTLY of the teacher — Codex Round
            # 2 Critical #1: ablation cells that skip the teacher (decay=0)
            # still need a matched phase schedule so LoRA unfreeze timing is
            # the same across C1..C5.
            if self._backbone_unfreeze_step > 0:
                self._freeze_lora_params()

        if loss_weights is not None:
            # Filter out losses that are not used
            assert (
                loss_weights.keys() == loss_fns.keys()
            ), f"Loss weight keys {loss_weights.keys()} != {loss_fns.keys()}"
            # loss_fns_filtered = {k: loss for k, loss in loss_fns.items() if loss_weights[k] != 0.0}
            # loss_weights_filtered = {
            #     k: loss for k, loss in loss_weights.items() if loss_weights[k] != 0.0
            # }
            self.loss_fns = nn.ModuleDict(loss_fns)
            self.loss_weights = loss_weights
        else:
            self.loss_fns = nn.ModuleDict(loss_fns)
            self.loss_weights = {}

        self.mask_resizers = mask_resizers if mask_resizers else {}
        self.mask_resizers["segmentation"] = modules.Resizer(
            video_inputs=input_type == "video", resize_mode="nearest-exact"
        )
        self.mask_soft_to_hard = modules.SoftToHardMask()
        self.train_metrics = torch.nn.ModuleDict(train_metrics)
        self.val_metrics = torch.nn.ModuleDict(val_metrics)

        self.visualize = visualize
        if visualize:
            assert visualize_every_n_steps is not None
        self.visualize_every_n_steps = visualize_every_n_steps
        if isinstance(masks_to_visualize, str):
            masks_to_visualize = [masks_to_visualize]
        for key in masks_to_visualize:
            if key not in ("decoder", "grouping", "dynamics_predictor"):
                raise ValueError(f"Unknown mask type {key}. Should be `decoder` or `grouping`.")
        self.mask_keys_to_visualize = [f"{key}_masks" for key in masks_to_visualize]

        if input_type == "image":
            self.input_key = "image"
            self.expected_input_dims = 4
        elif input_type == "video":
            self.input_key = "video"
            self.expected_input_dims = 5
        else:
            raise ValueError(f"Unknown input type {input_type}. Should be `image` or `video`.")

        self.target_encoder_input_key = (
            target_encoder_input if target_encoder_input else self.input_key
        )

    def configure_optimizers(self):
        modules = {
            "initializer": self.initializer,
            "encoder": self.encoder,
            "processor": self.processor,
            "decoder": self.decoder,
        }
        if self.dynamics_predictor:
            modules["dynamics_predictor"] = self.dynamics_predictor
        # Open-set head is an opt-in submodule; only register with the
        # optimizer when it is actually present.
        if self.open_set_head is not None:
            modules["open_set_head"] = self.open_set_head
        return self.optimizer_builder(modules)

    # --- GSRS teacher-snapshot plumbing (proposal §3.12). -------------------

    # Gate G2 threshold table: the snapshot-time teacher must reach this
    # FG-ARI before we freeze it (else the replay teacher drifts us into a
    # bad-identity regime). Threshold = 0.90 × published GCv1 matched-compute
    # FG-ARI on the same dataset. Datasets not listed here trip a loud
    # "no-threshold-configured" error at snapshot time — no silent fallback.
    #
    # Source: ``refine-logs/EXPERIMENT_PLAN_2026_04_15.md`` §(baselines) —
    # GCv1 matched-compute FG-ARI is 0.622 on MOVi-D, 0.323 on YT-VIS 2021.
    GSRS_G2_FG_ARI_THRESHOLDS: Dict[str, float] = {
        "movi_d":   0.90 * 0.622,   # = 0.560
        "ytvis2021": 0.90 * 0.323,  # = 0.291
    }

    def freeze_teacher_at_step(
        self,
        step: int,
        ckpt_path: Optional[str] = None,
    ) -> None:
        """Snapshot (encoder + initializer + processor) into a frozen teacher.

        Called at the step specified by ``trainer.teacher_snapshot_step``
        (propagated into ``self.teacher_snapshot_step`` at build time) via
        :meth:`on_train_batch_start`. The snapshot is a deep-copy of the
        current encoder + initializer + processor modules with
        ``requires_grad=False`` on every parameter; it is stored on
        ``self.teacher`` and NOT registered for the optimizer. The decoder
        and the open-set head are intentionally excluded — the replay
        loss only consumes teacher slot trajectories, and the renderer
        handles the decoding pathway during replay.

        The call is idempotent: re-firing at the same step is a no-op.
        Passing ``ckpt_path`` writes the teacher state dict to disk for
        downstream inspection (§3.12 permits a renderer re-certification
        step to validate the snapshot before proceeding).
        """
        if self._teacher_snapshot_done:
            return
        # Deep-copy the three sub-networks that define the teacher's slot
        # trajectory pipeline. ``deepcopy`` preserves the existing buffers
        # and parameter init exactly, mirroring the GCv2 pretrained state.
        teacher_bundle = nn.ModuleDict({
            "encoder": deepcopy(self.encoder),
            "initializer": deepcopy(self.initializer),
            "processor": deepcopy(self.processor),
        })
        # Re-bind TimmExtractor forward hooks on the copied encoder.
        # PyTorch's ``deepcopy`` preserves the hook handle registry on each
        # module but leaves the hook callback's closure pointing at the
        # ORIGINAL TimmExtractor's ``feature_outputs`` dict — so without
        # this rebinding the teacher's forward would silently write into
        # the student's feature buffer (observed at integration time when
        # the teacher encoder returned ``None`` for its main feature key).
        self._rebind_timm_hooks(teacher_bundle["encoder"])
        # Freeze all teacher parameters — no grad flow through the snapshot.
        for p in teacher_bundle.parameters():
            p.requires_grad = False
        teacher_bundle.eval()
        self.teacher = teacher_bundle
        self._teacher_snapshot_done = True
        if ckpt_path is not None:
            torch.save(
                {"state_dict": teacher_bundle.state_dict(), "step": int(step)},
                ckpt_path,
            )

    @staticmethod
    def _rebind_timm_hooks(encoder_module: nn.Module) -> None:
        """Walk ``encoder_module`` and re-register any TimmExtractor forward
        hooks so they write into the COPY's ``feature_outputs`` dict, not
        the original's.

        This fixes a latent bug in ``deepcopy(encoder)`` where the hooks'
        Python closures still reference the source ``self`` — a direct
        consequence of ``deepcopy`` preserving hook handles but not
        re-creating the closure-captured attribute bindings.

        We only touch modules of type ``TimmExtractor``; everything else is
        a no-op. We do NOT import ``TimmExtractor`` at the top of the file
        to avoid paying the timm import cost on every model build;
        instead we duck-type by attribute signature.
        """
        for module in encoder_module.modules():
            # Duck-type check: TimmExtractor exposes `feature_outputs`,
            # `features` (list of feature names), `model` (timm backbone),
            # and `hooks` (list of handles).
            if (
                hasattr(module, "feature_outputs")
                and hasattr(module, "features")
                and hasattr(module, "hooks")
                and hasattr(module, "model")
            ):
                if getattr(module, "features", None) is None:
                    continue
                # Remove any stale handles inherited from deepcopy.
                for h in list(getattr(module, "hooks", [])):
                    try:
                        h.remove()
                    except Exception:
                        pass
                module.hooks = []
                module.feature_outputs = {}
                # Also strip any _forward_hooks left behind on submodules
                # by the deepcopy (they still reference the original's
                # feature_outputs dict).
                for sub in module.model.modules():
                    if hasattr(sub, "_forward_hooks"):
                        # Keep only hooks that are NOT our TimmExtractor's
                        # (we can't distinguish cleanly, so clear all —
                        # TimmExtractor's hooks are the only ones it
                        # registers on internal blocks).
                        sub._forward_hooks.clear()
                # Re-register using the same alias resolution that
                # ``TimmExtractor.__init__`` performs.
                FEATURE_ALIASES = module.FEATURE_ALIASES
                FEATURE_MAPPING = module.FEATURE_MAPPING

                def _make_hook(captured_dict, key):
                    def _h(_mod, _inp, out):
                        captured_dict[key] = out
                    return _h

                for feature_name in module.features:
                    target_name = feature_name
                    if feature_name in FEATURE_ALIASES:
                        target_name = FEATURE_ALIASES[feature_name]
                    parts = target_name.split(".")
                    target = module.model
                    for part in parts:
                        if part.isdigit():
                            target = target[int(part)]
                        else:
                            target = getattr(target, part)
                    key = FEATURE_MAPPING.get(target_name, feature_name)
                    handle = target.register_forward_hook(
                        _make_hook(module.feature_outputs, key)
                    )
                    module.hooks.append(handle)

    # ------------------------------------------------------------------ #
    #  TubeGram EMA teacher                                                #
    # ------------------------------------------------------------------ #

    def _init_ema_encoder(self) -> None:
        """Round 2 A1: snapshot full student pipeline into frozen teacher.

        Deep-copies encoder + initializer + processor + decoder at
        `_backbone_unfreeze_step` so the teacher captures Phase-1-end state.
        All copies are frozen (``requires_grad=False``) and put in ``eval()``
        mode. The teacher is re-used every forward to produce stable
        ``teacher_decoder_masks_soft`` (tube assignments) and
        ``ema_backbone_features`` (stable covariance target) for
        ``TubeGramLoss`` / ``ChiBoostLoss``.

        Clears (a) ``feature_outputs`` dicts and (b) ``_forward_hooks``
        OrderedDicts on every encoder submodule before deepcopy because
        hook closures retain non-leaf activation tensors that break
        ``torch.Tensor.__deepcopy__``.
        """
        if self._ema_encoder is not None:
            return
        saved_outputs = {}
        saved_fwd_hooks = {}
        saved_hook_list = {}
        for name, m in self.encoder.named_modules():
            if hasattr(m, "feature_outputs") and isinstance(m.feature_outputs, dict):
                saved_outputs[name] = m.feature_outputs
                m.feature_outputs = {}
            if hasattr(m, "_forward_hooks") and len(m._forward_hooks) > 0:
                saved_fwd_hooks[name] = m._forward_hooks
                m._forward_hooks = type(m._forward_hooks)()
            if hasattr(m, "hooks") and isinstance(m.hooks, list) and len(m.hooks) > 0:
                saved_hook_list[name] = m.hooks
                m.hooks = []
        try:
            with torch.no_grad():
                self._ema_encoder = deepcopy(self.encoder)
                self._ema_initializer = deepcopy(self.initializer)
                self._ema_processor = deepcopy(self.processor)
                self._ema_decoder = deepcopy(self.decoder)
        finally:
            for name, m in self.encoder.named_modules():
                if name in saved_outputs:
                    m.feature_outputs = saved_outputs[name]
                if name in saved_fwd_hooks:
                    m._forward_hooks = saved_fwd_hooks[name]
                if name in saved_hook_list:
                    m.hooks = saved_hook_list[name]
        self._rebind_timm_hooks(self._ema_encoder)
        for module in (
            self._ema_encoder,
            self._ema_initializer,
            self._ema_processor,
            self._ema_decoder,
        ):
            for p in module.parameters():
                p.requires_grad = False
            module.eval()

    @torch.no_grad()
    def _update_ema_encoder(self) -> None:
        """No-op since Round 2 A1: teacher is a frozen snapshot, not EMA.

        Retained as a named method so callers (``on_train_batch_start``)
        don't need to branch. If future work wants to reintroduce a moving
        teacher on top of the snapshot, reimplement here.
        """
        return

    def _freeze_lora_params(self) -> None:
        """Phase 1 (before unfreeze step): force LoRA adapter params to
        ``requires_grad=False`` so the backbone is effectively frozen while
        the slot attention + decoder stabilise. Non-LoRA params are already
        frozen by ``apply_lora`` (see ``slotcontrast/modules/lora.py``).
        """
        for name, p in self.encoder.named_parameters():
            if "lora_" in name:
                p.requires_grad = False

    def _unfreeze_lora_params(self) -> None:
        """Phase 2 transition: flip LoRA params back to trainable at the
        configured ``_backbone_unfreeze_step``. Opposite of
        ``_freeze_lora_params``.
        """
        for name, p in self.encoder.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

    def _run_gate_g2_fg_ari_check(self, max_clips: int = 250) -> float:
        """Gate G2: run validation on a capped subset before teacher snapshot.

        Returns the measured ``val/fg_ari`` as a float. The caller compares
        against a dataset-specific threshold and raises if below. See
        proposal §3.12 and ``GSRS_G2_FG_ARI_THRESHOLDS``.

        The check uses the currently-configured ``val_metrics['fg_ari']``
        instance. If no ``fg_ari`` metric is registered, we raise rather
        than silently skipping the gate (no fallback per project rule).
        """
        if "fg_ari" not in self.val_metrics:
            raise RuntimeError(
                "Gate G2: model.val_metrics has no 'fg_ari' entry. "
                "Gate G2 requires an FG-ARI val metric; register one in "
                "the config's `val_metrics` block before enabling the "
                "GSRS teacher snapshot."
            )
        fg_ari_metric = self.val_metrics["fg_ari"]

        if self.trainer is None or not hasattr(self.trainer, "datamodule"):
            raise RuntimeError(
                "Gate G2: trainer / datamodule not attached. The gate is "
                "meant to fire inside on_train_batch_start with Lightning's "
                "trainer already initialised."
            )
        val_loader = self.trainer.datamodule.val_dataloader()

        # Save training mode and switch to eval for the gate.
        was_training = self.training
        self.eval()
        fg_ari_metric.reset()

        device = next(self.parameters()).device
        n_clips = 0
        with torch.no_grad():
            for batch in val_loader:
                if n_clips >= max_clips:
                    break
                # Strip padding rows if present — mirrors validation_step.
                if "batch_padding_mask" in batch:
                    batch = self._remove_padding(batch, batch["batch_padding_mask"])
                    if batch is None:
                        continue
                # Move to device.
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                outputs = self.forward(batch)
                aux_outputs = self.aux_forward(batch, outputs)
                fg_ari_metric.update(**{**batch, **outputs, **aux_outputs})
                n_clips += outputs["batch_size"]

        # Restore mode.
        if was_training:
            self.train()

        val_fg_ari = fg_ari_metric.compute()
        fg_ari_metric.reset()
        # fg_ari may be a tensor scalar or dict; normalize to float.
        if isinstance(val_fg_ari, dict):
            # Some ARI metrics report a dict of components; pick the
            # 'fg_ari' / 'overall' sub-key if present, else raise.
            if "fg_ari" in val_fg_ari:
                val_fg_ari = val_fg_ari["fg_ari"]
            elif "overall" in val_fg_ari:
                val_fg_ari = val_fg_ari["overall"]
            else:
                raise RuntimeError(
                    f"Gate G2: fg_ari metric returned a dict "
                    f"{list(val_fg_ari.keys())} — no recognisable scalar."
                )
        return float(val_fg_ari)

    def _gate_g2_check_and_raise(self, dataset_name: str) -> None:
        """Run Gate G2, compare to threshold, raise on failure.

        ``dataset_name`` must be a key in ``GSRS_G2_FG_ARI_THRESHOLDS``.
        No silent fallback: an unknown dataset raises.
        """
        if dataset_name not in self.GSRS_G2_FG_ARI_THRESHOLDS:
            raise ValueError(
                f"Gate G2: no FG-ARI threshold configured for dataset "
                f"{dataset_name!r}. Known datasets: "
                f"{list(self.GSRS_G2_FG_ARI_THRESHOLDS.keys())}. Add an "
                f"entry to ObjectCentricModel.GSRS_G2_FG_ARI_THRESHOLDS "
                f"or set `model.gsrs_g2_dataset` to a known key."
            )
        threshold = self.GSRS_G2_FG_ARI_THRESHOLDS[dataset_name]
        val_fg_ari = self._run_gate_g2_fg_ari_check()
        if val_fg_ari < threshold:
            raise RuntimeError(
                f"Gate G2 failed: teacher FG-ARI {val_fg_ari:.3f} < "
                f"threshold {threshold:.3f} "
                f"(0.90 × published GCv1 matched-compute FG-ARI on "
                f"{dataset_name!r}). Refusing to snapshot a bad teacher."
            )
        print(
            f"[GSRS Gate G2] teacher FG-ARI={val_fg_ari:.3f} >= "
            f"threshold {threshold:.3f} on {dataset_name!r}; proceeding "
            f"with teacher snapshot.",
            flush=True,
        )

    def on_train_batch_start(self, batch, batch_idx):
        """Lightning hook — triggers the teacher snapshot at the configured step.

        This is the only automatic trigger for ``freeze_teacher_at_step``.
        External callers (e.g. a custom callback) may still invoke the
        method directly; idempotence guarantees no double-snapshot.

        GSRS Gate G2: before deep-copying the student into the teacher,
        run val on a capped subset and require
        ``val/fg_ari ≥ 0.90 × published GCv1 matched-compute FG-ARI`` on
        the configured dataset (see ``GSRS_G2_FG_ARI_THRESHOLDS``). On
        failure we raise ``RuntimeError`` rather than snapshot a
        bad-identity teacher that would poison the replay branch.
        """
        if self.teacher_snapshot_step > 0 and not self._teacher_snapshot_done:
            # Global step is 0-indexed; fire at the *first* step whose index
            # is ≥ the configured snapshot step to avoid off-by-one issues
            # when the train loop resumes from a checkpoint mid-warmup.
            step = int(self.trainer.global_step)
            if step >= self.teacher_snapshot_step:
                # GSRS Gate G2 check (§3.12). The dataset name is read off
                # the trainer's datamodule shard path (heuristic) or a
                # ``model.gsrs_g2_dataset`` attribute if set explicitly.
                dataset_name = getattr(self, "gsrs_g2_dataset", None)
                if dataset_name is None:
                    dataset_name = self._infer_gsrs_dataset_name()
                self._gate_g2_check_and_raise(dataset_name)
                self.freeze_teacher_at_step(step=step, ckpt_path=None)

        # Phase schedule — LoRA freeze/unfreeze (Round 1 Critical #2/#3,
        # Round 2 Codex Critical #1). Runs INDEPENDENTLY of teacher
        # snapshot so ablation cells that skip the teacher (`decay == 0`)
        # still get a matched schedule.
        if self._backbone_unfreeze_step > 0:
            step = int(self.trainer.global_step)
            if step >= self._backbone_unfreeze_step and not self._backbone_unfrozen:
                self._unfreeze_lora_params()
                self._backbone_unfrozen = True
                if not getattr(self, "_init_hash_logged_unfreeze", False):
                    self._log_init_hash("post_unfreeze")
                    self._init_hash_logged_unfreeze = True
                # Round-24 Fix #3: snapshot the (now-warmed-up) student
                # output_transform so the projected-anchor loss can compare
                # student.encoder.features against a stable
                # snapshot(target_encoder.backbone_features) target.
                if (
                    self.use_projected_anchor
                    and self._snapshot_output_transform is None
                    and self.target_encoder is not None
                ):
                    enc_module = getattr(self.encoder, "module", self.encoder)
                    if getattr(enc_module, "output_transform", None) is not None:
                        self._snapshot_output_transform = deepcopy(
                            enc_module.output_transform
                        )
                        for p in self._snapshot_output_transform.parameters():
                            p.requires_grad_(False)
                        self._snapshot_output_transform.eval()

        # Teacher snapshot — gated on `_ema_decay > 0`. Fires exactly once
        # at/after `_backbone_unfreeze_step`. Post-snapshot the call is
        # idempotent (`_ema_encoder is not None` → skip). `_update_ema_encoder`
        # is a no-op retained for future variants.
        if self._ema_decay > 0 and self._backbone_unfreeze_step > 0:
            step = int(self.trainer.global_step)
            if step >= self._backbone_unfreeze_step and self._ema_encoder is None:
                self._init_ema_encoder()

    def _infer_gsrs_dataset_name(self) -> str:
        """Heuristically read the dataset name from the attached datamodule.

        We peek at ``self.trainer.datamodule.train_shards`` (a list of
        shard paths) and look for any recognised dataset token in the
        first shard. Raises if nothing matches (no fallback).
        """
        try:
            dm = self.trainer.datamodule
            shards = getattr(dm, "train_shards", None) or getattr(
                dm, "val_shards", None
            )
            if not shards:
                raise RuntimeError("datamodule exposes no shard list")
            first = str(shards[0] if isinstance(shards, (list, tuple)) else shards)
        except Exception as exc:  # pragma: no cover — defensive
            raise RuntimeError(
                f"Gate G2: failed to infer dataset from datamodule: {exc}"
            )
        for key in self.GSRS_G2_FG_ARI_THRESHOLDS:
            if key in first:
                return key
        raise RuntimeError(
            f"Gate G2: cannot identify dataset from shard path {first!r}; "
            f"set ``model.gsrs_g2_dataset`` explicitly in the config."
        )

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        encoder_input = inputs[self.input_key]  # batch [x n_frames] x n_channels x height x width
        assert encoder_input.ndim == self.expected_input_dims
        batch_size = len(encoder_input)

        # Pack camera_data from individual keys if present (for 3D positional embedding)
        camera_data = None
        if "depths" in inputs and "intrinsics" in inputs and "extrinsics" in inputs:
            camera_data = {
                "depth": inputs["depths"],
                "intrinsics": inputs["intrinsics"],
                "extrinsics": inputs["extrinsics"],
            }
        encoder_output = self.encoder(encoder_input, camera_data=camera_data)
        features = encoder_output["features"]

        # Round-24 evolution: optional frozen-feature init source.
        # When self.initializer_feature_source == "target_encoder", run the
        # frozen target_encoder once here (no_grad) and use its
        # backbone_features for greedy/saliency slot init. The forward output
        # caches the result so get_targets() can reuse it without a second
        # forward pass.
        target_encoder_output: Optional[Dict[str, torch.Tensor]] = None
        need_target_forward = (
            self.initializer_feature_source == "target_encoder"
            or (self.use_projected_anchor and self.target_encoder is not None)
        )
        if need_target_forward:
            target_input = inputs.get(
                self.target_encoder_input_key, encoder_input
            )
            with torch.no_grad():
                target_encoder_output = self.target_encoder(target_input)
            # Round-24 Fix #3: project frozen target features through the
            # snapshot of `output_transform` (taken at backbone_unfreeze_step
            # in `on_train_batch_start`) to produce a stable 128d anchor for
            # the student-encoder features. Pre-snapshot the field is absent;
            # the anchor loss should be configured with weight=0 until
            # backbone_unfreeze_step or guard on key presence.
            if self.use_projected_anchor:
                if (
                    self._snapshot_output_transform is not None
                    and "backbone_features" in target_encoder_output
                ):
                    with torch.no_grad():
                        proj = self._snapshot_output_transform(
                            target_encoder_output["backbone_features"]
                        )
                    target_encoder_output["projected_features"] = proj.detach()
                else:
                    # Pre-snapshot placeholder — student.encoder.features
                    # cloned so the projected-anchor MSE is trivially 0 until
                    # the snapshot is taken at backbone_unfreeze_step. The
                    # ramp factor in compute_loss also zeroes the weight in
                    # this regime, so this is belt-and-suspenders.
                    target_encoder_output["projected_features"] = features.detach()

        # Pull per-sample depth BEFORE the initializer so depth-aware initializers
        # (e.g. DepthEdgeFeatureInit) can consume it. Depth-required initializers
        # must set `requires_depth = True` on their class; we hard-raise here
        # instead of silently falling back (user's no-fallback directive).
        # Only pass `depth=` when the initializer opts in — base GreedyFeatureInit
        # etc. don't accept depth kwarg and would TypeError otherwise.
        init_depth = inputs.get("depth")
        init_requires_depth = getattr(self.initializer, "requires_depth", False)
        if init_requires_depth and init_depth is None:
            raise RuntimeError(
                f"Initializer {type(self.initializer).__name__} has "
                f"requires_depth=True but `inputs['depth']` is missing. Configure "
                f"`dataset.depth_cache_dir` and add `depth` to the pipeline keys."
            )
        init_kwargs = {"depth": init_depth} if init_requires_depth else {}

        # Use backbone features for initialization (more stable early in training)
        # Round-24: when initializer_feature_source=="target_encoder", route
        # the saliency input through the frozen target_encoder backbone
        # features (immune to LoRA / output_transform drift). The student
        # encoder still drives slot attention and the decoder downstream.
        if self.initializer_feature_source == "target_encoder":
            assert target_encoder_output is not None
            init_features = target_encoder_output["backbone_features"]
            init_output = self.initializer(
                batch_size=batch_size, features=init_features, **init_kwargs
            )
            encoder_module = getattr(self.encoder, "module", self.encoder)
            if isinstance(init_output, tuple):
                raw_slots, n_objects, existence_mask = init_output
                slots_initial = encoder_module.output_transform(raw_slots)
            else:
                slots_initial = encoder_module.output_transform(init_output)
                n_objects, existence_mask = None, None
        elif self.use_backbone_features and "backbone_features" in encoder_output:
            backbone_features = encoder_output["backbone_features"]
            init_output = self.initializer(batch_size=batch_size, features=backbone_features, **init_kwargs)
            # Get output_transform from encoder (handle MapOverTime wrapper for video)
            encoder_module = getattr(self.encoder, 'module', self.encoder)
            # Handle both single tensor and tuple output from initializer
            if isinstance(init_output, tuple):
                raw_slots, n_objects, existence_mask = init_output
                slots_initial = encoder_module.output_transform(raw_slots)
            else:
                slots_initial = encoder_module.output_transform(init_output)
                n_objects, existence_mask = None, None
        else:
            init_output = self.initializer(batch_size=batch_size, features=features, **init_kwargs)
            # Handle both single tensor and tuple output from initializer
            if isinstance(init_output, tuple):
                slots_initial, n_objects, existence_mask = init_output
            else:
                slots_initial = init_output
                n_objects, existence_mask = None, None
        
        # Pull optical flow from the batch if present (v9 motion branch).
        # Accept either `forward_flow` (Kubric/RAFT convention) or a generic
        # `flow` key. Only passed through when `input_type == "video"` because
        # ScanOverTime is the only processor that declares a `flow` kwarg.
        flow = inputs.get("forward_flow", inputs.get("flow"))
        depth = inputs.get("depth")
        extra_processor_kwargs = {}
        if flow is not None and self.input_key == "video":
            extra_processor_kwargs["flow"] = flow
        if depth is not None and self.input_key == "video":
            extra_processor_kwargs["depth"] = depth

        # Pass existence_mask through processor for variable slot support
        processor_output = self.processor(
            slots_initial, features,
            existence_mask=existence_mask, **extra_processor_kwargs,
        )
        slots = processor_output["state"]
        
        # Use processor output existence_mask if available (from memory matcher)
        out_existence_mask = processor_output.get("existence_mask", existence_mask)
        decoder_output = self.decoder(slots, existence_mask=out_existence_mask)

        outputs = {
            "batch_size": batch_size,
            "encoder": encoder_output,
            "processor": processor_output,
            "decoder": decoder_output,
        }
        if target_encoder_output is not None:
            # Cache the early target_encoder forward so get_targets() can reuse
            # it for featrec without re-running the frozen backbone.
            outputs["target_encoder"] = target_encoder_output

        # TubeGram: lift soft decoder masks and EMA backbone features.
        if "masks" in decoder_output:
            outputs["decoder_masks_soft"] = decoder_output["masks"]
        # D3 MOCSP: lift encoder-emitted input_mask so MOCSPLoss can locate
        # held-out patches via the ``aux_keys`` routing in ``compute_loss``.
        if isinstance(encoder_output, dict) and "input_mask" in encoder_output:
            outputs["input_mask"] = encoder_output["input_mask"]
        # Round 2 A1: run the snapshot teacher pipeline to produce
        # `teacher_decoder_masks_soft` (stable tube assignments) and
        # `ema_backbone_features` (stable covariance target). Only on training
        # steps — val does not need the teacher (the ramped losses are only
        # logged, not optimised, during validation and the teacher forward
        # would double val wall-clock). The teacher is frozen + eval() so
        # gradients never leak back through it.
        if self._ema_decay > 0 and self.training and self._ema_encoder is not None:
            with torch.no_grad():
                for m in (
                    self._ema_encoder,
                    self._ema_initializer,
                    self._ema_processor,
                    self._ema_decoder,
                ):
                    m.eval()
                teacher_enc_out = self._ema_encoder(
                    encoder_input, camera_data=camera_data
                )
                teacher_features = teacher_enc_out["features"]
                teacher_backbone = teacher_enc_out.get("backbone_features")

                if self.use_backbone_features and teacher_backbone is not None:
                    teacher_init_out = self._ema_initializer(
                        batch_size=batch_size,
                        features=teacher_backbone,
                        **init_kwargs,
                    )
                    teacher_encoder_module = getattr(
                        self._ema_encoder, "module", self._ema_encoder
                    )
                    if isinstance(teacher_init_out, tuple):
                        t_raw_slots, _t_n_objects, _t_existence = teacher_init_out
                        teacher_slots_initial = teacher_encoder_module.output_transform(
                            t_raw_slots
                        )
                    else:
                        teacher_slots_initial = teacher_encoder_module.output_transform(
                            teacher_init_out
                        )
                else:
                    teacher_init_out = self._ema_initializer(
                        batch_size=batch_size,
                        features=teacher_features,
                        **init_kwargs,
                    )
                    if isinstance(teacher_init_out, tuple):
                        teacher_slots_initial, _t_n_objects, _t_existence = (
                            teacher_init_out
                        )
                    else:
                        teacher_slots_initial = teacher_init_out

                teacher_proc_out = self._ema_processor(
                    teacher_slots_initial,
                    teacher_features,
                    existence_mask=existence_mask,
                    **extra_processor_kwargs,
                )
                teacher_slots = teacher_proc_out["state"]
                t_out_existence = teacher_proc_out.get(
                    "existence_mask", out_existence_mask
                )
                teacher_dec_out = self._ema_decoder(
                    teacher_slots, existence_mask=t_out_existence
                )
                if "masks" in teacher_dec_out:
                    outputs["teacher_decoder_masks_soft"] = teacher_dec_out["masks"]
                if teacher_backbone is not None:
                    outputs["ema_backbone_features"] = teacher_backbone

        # Add variable slot info if available (from GreedyFeatureInitV2)
        if n_objects is not None:
            outputs["n_objects"] = n_objects
        if out_existence_mask is not None:
            outputs["existence_mask"] = out_existence_mask

        # APP: lift amodal particle outputs to top-level for loss consumption.
        for app_key in (
            "visible_masks", "amodal_masks", "amodal_transported",
            "occlusion_plausibility", "particle_states", "prev_particle_states",
        ):
            if app_key in processor_output:
                outputs[app_key] = processor_output[app_key]

        # Slot-Drop-Recover (Idea #012): lift per-frame-stacked snapshot and
        # drop mask from the processor output to the top-level outputs dict so
        # that `compute_loss` can forward them as kwargs (`pre_drop_slots`,
        # `drop_mask`) to `SlotDropRecoverLoss` via its `aux_keys` declaration.
        # No-op (shape-stable zeros) when `slot_drop_prob=0`, so downstream
        # consumers that don't declare these aux keys are unaffected.
        if "pre_drop_slots" in processor_output:
            outputs["pre_drop_slots"] = processor_output["pre_drop_slots"]
        if "drop_mask" in processor_output:
            outputs["drop_mask"] = processor_output["drop_mask"]

        # Flow-Consistency Mask Regularizer (Idea T3-02): lift forward flow
        # (if the batch carries it) and Hungarian match indices (if the
        # predictor exposes them) to the top-level outputs dict so that
        # `compute_loss` can forward them as kwargs (`forward_flow`,
        # `hungarian_match_indices`) to `FlowConsistencyMaskLoss` via its
        # `aux_keys` declaration. No-op when the corresponding inputs are
        # absent — other losses that don't declare these aux keys are
        # unaffected.
        if flow is not None and self.input_key == "video":
            outputs["forward_flow"] = flow
        if "hungarian_match_indices" in processor_output:
            outputs["hungarian_match_indices"] = processor_output["hungarian_match_indices"]

        # GSRS open-set head (proposal §3.1(a), §3.13). Runs on the
        # student slot state whenever the head is configured; emits
        # `[..., 4]` softmax probs over {source, dormant, null, born}.
        # Gradients flow back through `slots`, by design. No-op when the
        # head is not configured.
        if self.open_set_head is not None:
            outputs["open_set_probs"] = self.open_set_head(slots)

        # GSRS replay branch (proposal §3.9, §3.12). Activated only when
        # the batch carries an explicit `replay_family` label — i.e. this
        # is a replay sample, not a real-clip batch. All behaviour is
        # gated on that key; on real-clip batches the path below is a
        # strict no-op and does NOT perturb other consumers of
        # `self.forward`. We hard-raise on partial replay inputs: a
        # batch that sets `replay_family` but omits `replay_target_
        # trajectories` or `replay_event_flags` is a configuration bug
        # (per the project-wide no-fallback directive).
        if "replay_family" in inputs:
            replay_family = inputs["replay_family"]
            replay_targets = inputs.get("replay_target_trajectories")
            replay_events = inputs.get("replay_event_flags")
            if replay_targets is None or replay_events is None:
                raise ValueError(
                    "GSRS replay branch: `inputs['replay_family']` is set but "
                    "`replay_target_trajectories` and/or `replay_event_flags` "
                    "are missing. Plumb all three (family, target trajectories, "
                    "event flags) together — no silent fallback."
                )
            outputs["replay_family"] = replay_family
            outputs["replay_target_trajectories"] = replay_targets
            outputs["replay_event_flags"] = replay_events
            # Run the frozen teacher if it has been snapshotted (§3.12).
            # Teacher slots are lifted onto outputs for downstream
            # consistency losses; if the teacher hasn't been snapshotted
            # yet (Phase 1 warmup), we surface a loud error rather than
            # silently feeding the student slots as their own teacher.
            if self.teacher is None:
                raise RuntimeError(
                    "GSRS replay sample arrived but `self.teacher` is None — "
                    "the teacher snapshot has not yet been taken. Either "
                    "delay replay-batch emission until after the configured "
                    "`teacher_snapshot_step`, or invoke "
                    "`model.freeze_teacher_at_step(step=...)` explicitly."
                )
            with torch.no_grad():
                teacher_enc = self.teacher["encoder"](encoder_input, camera_data=camera_data)
                teacher_features = teacher_enc["features"]
                teacher_init_out = self.teacher["initializer"](
                    batch_size=batch_size, features=teacher_features, **init_kwargs
                )
                if isinstance(teacher_init_out, tuple):
                    teacher_slots_init = teacher_init_out[0]
                else:
                    teacher_slots_init = teacher_init_out
                # Run teacher processor with same per-video kwargs that
                # the student saw (flow / depth), to keep the two paths
                # comparable when the replay branch consumes them.
                teacher_proc_out = self.teacher["processor"](
                    teacher_slots_init, teacher_features,
                    existence_mask=existence_mask, **extra_processor_kwargs,
                )
                outputs["teacher_slots"] = teacher_proc_out["state"]

        # Cycle/Temporal Cross-Consistency: Re-slot the reconstructed features
        # When window=0, this is same-frame cycle consistency
        # When window>0, this includes cross-frame temporal consistency
        if self.use_cycle_consistency:
            cycle_slots, cycle_targets = self._compute_cycle_slots(
                processor_output, decoder_output, 
                window=self.temporal_cross_window,
                mode=self.temporal_cross_mode
            )
            outputs["processor"]["cycle_slots"] = cycle_slots
            outputs["processor"]["cycle_targets"] = cycle_targets

        if self.dynamics_predictor:
            outputs["dynamics_predictor"] = self.dynamics_predictor(slots)
            predicted_slots = outputs["dynamics_predictor"].get("next_state")
            decoded_predicted_slots = self.decoder(predicted_slots)
            decoded_predicted_slots = {
                f"predicted_{key}": value for key, value in decoded_predicted_slots.items()
            }
            outputs["decoder"].update(decoded_predicted_slots)

        outputs["targets"] = self.get_targets(inputs, outputs)

        return outputs

    def _compute_cycle_slots(
        self, processor_output: Dict[str, Any], decoder_output: Dict[str, Any], 
        window: int = 0, mode: str = "both"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cycle/temporal cross-consistency slots.
        
        When window=0: Same-frame cycle consistency (queries from t, features from t)
        When window>0: Temporal cross-consistency with mode:
            - "both": queries from [t-window, t+window]
            - "backward": queries from [t-window, t]
            - "forward": queries from [t, t+window]
        
        Returns both cycle slots and detached target slots.
        """
        recon_features = decoder_output["reconstruction"]  # [B, T, P, D_feat] or [B, P, D_feat]
        initial_queries = processor_output["initial_queries"]  # [B, T, K, D_slot] or [B, K, D_slot]
        real_slots = processor_output["corrector"]["slots"]  # [B, T, K, D_slot] or [B, K, D_slot]
        
        is_video = recon_features.ndim == 4
        
        if is_video:
            B, T, P, D_feat = recon_features.shape
            _, _, K, D_slot = initial_queries.shape
            
            # Ensure window size is valid
            assert 0 <= window <= T - 1, f"Window size {window} must be in range [0, {T - 1}]"
            assert mode in ("both", "backward", "forward"), f"Mode must be 'both', 'backward', or 'forward', got '{mode}'"
            
            # Transform reconstructed features to slot space
            output_transform = self.encoder.module.output_transform
            recon_flat = recon_features.flatten(0, 1)  # [B*T, P, D_feat]
            if output_transform is not None:
                recon_transformed = output_transform(recon_flat).view(B, T, P, -1)
            else:
                recon_transformed = recon_flat.view(B, T, P, -1)
            
            if window > 0:
                # Temporal cross-consistency: random sampling within window based on mode
                if mode == "both":
                    # Sample from [t-window, t+window]
                    offsets = torch.randint(-window, window + 1, (B, T), device=recon_features.device)
                elif mode == "backward":
                    # Sample from [t-window, t]
                    offsets = torch.randint(-window, 1, (B, T), device=recon_features.device)
                else:  # mode == "forward"
                    # Sample from [t, t+window]
                    offsets = torch.randint(0, window + 1, (B, T), device=recon_features.device)
                
                j_indices = torch.arange(T, device=recon_features.device).unsqueeze(0).expand(B, T)
                i_indices = (j_indices + offsets).clamp(0, T - 1)
                
                # Gather queries from time i
                i_expanded = i_indices.view(B, T, 1, 1).expand(B, T, K, D_slot)
                queries = torch.gather(initial_queries, dim=1, index=i_expanded)
            else:
                # Same-frame cycle consistency
                queries = initial_queries
            
            # Flatten and run slot attention
            queries_flat = queries.flatten(0, 1)
            features_flat = recon_transformed.flatten(0, 1)
            
            corrector = self.processor.module.corrector
            cycle_output = corrector(queries_flat, features_flat)
            cycle_slots = cycle_output["slots"].view(B, T, K, D_slot)
            target_slots = real_slots.detach()
        else:
            # Image case: always same-frame
            output_transform = self.encoder.output_transform
            if output_transform is not None:
                recon_features = output_transform(recon_features)
            
            corrector = self.processor.corrector
            cycle_output = corrector(initial_queries, recon_features)
            cycle_slots = cycle_output["slots"]
            target_slots = real_slots.detach()
        
        return cycle_slots, target_slots

    def process_masks(
        self,
        masks: torch.Tensor,
        inputs: Dict[str, Any],
        resizer: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Handle None or list of Nones (from merge_dict_trees when skip_corrector=True)
        if masks is None or (isinstance(masks, list) and all(m is None for m in masks)):
            return None, None, None

        if resizer is None:
            masks_for_vis = masks
            masks_for_vis_hard = self.mask_soft_to_hard(masks)
            masks_for_metrics_hard = masks_for_vis_hard
        else:
            masks_for_vis = resizer(masks, inputs[self.input_key])
            masks_for_vis_hard = self.mask_soft_to_hard(masks_for_vis)
            target_masks = inputs.get("segmentations")
            if target_masks is not None and masks_for_vis.shape[-2:] != target_masks.shape[-2:]:
                masks_for_metrics = resizer(masks, target_masks)
                masks_for_metrics_hard = self.mask_soft_to_hard(masks_for_metrics)
            else:
                masks_for_metrics_hard = masks_for_vis_hard

        return masks_for_vis, masks_for_vis_hard, masks_for_metrics_hard

    @torch.no_grad()
    def aux_forward(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute auxilliary outputs only needed for metrics and visualisations."""
        decoder_masks = outputs["decoder"].get("masks")
        decoder_masks, decoder_masks_hard, decoder_masks_metrics_hard = self.process_masks(
            decoder_masks, inputs, self.mask_resizers.get("decoder")
        )

        grouping_masks = outputs["processor"]["corrector"].get("masks")
        grouping_masks, grouping_masks_hard, grouping_masks_metrics_hard = self.process_masks(
            grouping_masks, inputs, self.mask_resizers.get("grouping")
        )

        aux_outputs = {}

        # APP: when visible_masks are available, use them for metrics instead
        # of the decoder masks (APP losses do not train the decoder, so its
        # masks carry no signal).
        if "visible_masks" in outputs:
            app_v = outputs["visible_masks"]
            _, app_hard, app_metrics_hard = self.process_masks(
                app_v, inputs, self.mask_resizers.get("decoder")
            )
            if app_hard is not None:
                decoder_masks_hard = app_hard
            if app_metrics_hard is not None:
                decoder_masks_metrics_hard = app_metrics_hard

        if decoder_masks is not None:
            aux_outputs["decoder_masks"] = decoder_masks
        if decoder_masks_hard is not None:
            aux_outputs["decoder_masks_vis_hard"] = decoder_masks_hard
        if decoder_masks_metrics_hard is not None:
            aux_outputs["decoder_masks_hard"] = decoder_masks_metrics_hard
        if grouping_masks is not None:
            aux_outputs["grouping_masks"] = grouping_masks
        if grouping_masks_hard is not None:
            aux_outputs["grouping_masks_vis_hard"] = grouping_masks_hard
        if grouping_masks_metrics_hard is not None:
            aux_outputs["grouping_masks_hard"] = grouping_masks_metrics_hard

        if self.dynamics_predictor:
            dynamics_predictor_masks = outputs["decoder"].get("predicted_masks")
            (
                dynamics_predictor_masks,
                dynamics_predictor_masks_hard,
                dynamics_predictor_masks_metrics_hard,
            ) = self.process_masks(
                dynamics_predictor_masks, inputs, self.mask_resizers.get("decoder")
            )
            if dynamics_predictor_masks is not None:
                aux_outputs["dynamics_predictor_masks"] = dynamics_predictor_masks
            if dynamics_predictor_masks_hard is not None:
                aux_outputs["dynamics_predictor_masks_vis_hard"] = dynamics_predictor_masks_hard
            if dynamics_predictor_masks_metrics_hard is not None:
                aux_outputs["dynamics_predictor_masks_hard"] = dynamics_predictor_masks_metrics_hard

        return aux_outputs

    def get_targets(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        if self.target_encoder and "target_encoder" not in outputs:
            target_encoder_input = inputs[self.target_encoder_input_key]
            assert target_encoder_input.ndim == self.expected_input_dims

            with torch.no_grad():
                encoder_output = self.target_encoder(target_encoder_input)

            outputs["target_encoder"] = encoder_output

        targets = {}
        for name, loss_fn in self.loss_fns.items():
            targets[name] = loss_fn.get_target(inputs, outputs)

        return targets

    def compute_loss(self, outputs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        existence_mask = outputs.get("existence_mask", None)

        for name, loss_fn in self.loss_fns.items():
            prediction = loss_fn.get_prediction(outputs)
            target = outputs["targets"][name]

            # Route per-loss mask by `mask_key` (round-24 L3) AND forward any
            # extra auxiliary tensors the loss declares via `aux_keys` (round-26
            # B1). Example: SlotDropRecoverLoss declares aux_keys=[
            # "pre_drop_slots","drop_mask"]; these flow from model forward via
            # `outputs[key]` to the loss kwargs.
            extra_kwargs = {}
            if hasattr(loss_fn, 'mask_key'):
                mask_key = getattr(loss_fn, 'mask_key', 'existence_mask')
                mask_value = outputs.get(mask_key, existence_mask)
                if mask_value is not None:
                    extra_kwargs[mask_key] = mask_value
            for aux_key in getattr(loss_fn, 'aux_keys', []):
                aux_value = outputs.get(aux_key, None)
                if aux_value is not None:
                    extra_kwargs[aux_key] = aux_value
            loss = loss_fn(prediction, target, **extra_kwargs)
            
            # Reduce all losses to scalars for logging
            if loss.ndim > 0:
                loss = loss.mean()
            losses[name] = loss

        # Phase-schedule weight ramp (Round 1 fix — Critical #2).
        # For losses in ``_ramped_loss_names`` the effective weight grows
        # linearly from 0 → configured weight over ``_loss_ramp_steps`` optimizer
        # steps *after* ``_backbone_unfreeze_step``. Before the unfreeze step the
        # ramp factor is 0, so TubeGram/ChiBoost do nothing while slot
        # attention + decoder stabilise. This replaces the old behaviour where
        # the hard confidence gate ``(conf - 1/K > 0.1)`` silently kept the
        # losses at 0 for the whole run.
        ramp_factor = 1.0
        step = int(self.trainer.global_step) if self.trainer is not None else 0
        if self._loss_ramp_steps > 0 or self._backbone_unfreeze_step > 0:
            if step < self._backbone_unfreeze_step:
                ramp_factor = 0.0
            elif self._loss_ramp_steps > 0:
                ramp_factor = min(
                    1.0,
                    float(step - self._backbone_unfreeze_step)
                    / float(self._loss_ramp_steps),
                )

        losses_weighted = []
        for name, loss in losses.items():
            weight = self.loss_weights.get(name, 1.0)
            if name in self._ramped_loss_names:
                weight = weight * ramp_factor
            losses_weighted.append(loss * weight)

        total_loss = torch.stack(losses_weighted).sum()

        return total_loss, losses

    def _log_init_hash(self, tag: str) -> None:
        """Log SHA256 of (sorted) named-parameter values to all attached
        WandB loggers AND a local JSON sidecar.

        Used by the Oral evidence pack to *prove* RNG-paired init: the same
        seed should produce the same hash for grouper / decoder / predictor
        across paired arms (frozen vs rescue), even though encoder params
        differ (rescue has LoRA layers). The JSON sidecar lives in the run's
        log directory so the audit script can reconstruct hashes even if
        WandB summaries are out of sync.

        Logged at ``on_train_start`` (tag ``t0``) and right after the
        backbone-unfreeze hook fires (tag ``post_unfreeze``).
        """
        import hashlib
        import json as _json
        try:
            full = hashlib.sha256()
            mod_hashes: Dict[str, "hashlib._Hash"] = {}
            shapes: Dict[str, List[int]] = {}
            for name, p in sorted(self.named_parameters(), key=lambda kv: kv[0]):
                arr = p.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
                full.update(arr)
                top = name.split(".", 1)[0]
                if top not in mod_hashes:
                    mod_hashes[top] = hashlib.sha256()
                mod_hashes[top].update(arr)
                shapes[name] = list(p.shape)
            payload: Dict[str, Any] = {
                f"init_hash/{tag}/full": full.hexdigest(),
            }
            for top, h in mod_hashes.items():
                payload[f"init_hash/{tag}/{top}"] = h.hexdigest()
            # WandB logger(s) — iterate self.loggers since multi-logger setups
            # break the singular self.logger attribute.
            loggers = []
            try:
                loggers = list(getattr(self, "loggers", []))
            except Exception:
                loggers = []
            if not loggers and getattr(self, "logger", None) is not None:
                loggers = [self.logger]
            for lg in loggers:
                try:
                    exp = getattr(lg, "experiment", None)
                    if exp is not None and hasattr(exp, "summary"):
                        for k, v in payload.items():
                            exp.summary[k] = v
                except Exception:
                    pass
            # JSON sidecar in the run log dir (independent of wandb).
            try:
                trainer = getattr(self, "trainer", None)
                logger0 = trainer.loggers[0] if trainer and trainer.loggers else None
                save_dir = None
                if logger0 is not None:
                    save_dir = getattr(logger0, "save_dir", None) or getattr(logger0, "experiment", None)
                    if save_dir is not None and not isinstance(save_dir, (str, bytes)):
                        save_dir = getattr(save_dir, "dir", None)
                if save_dir is None and trainer is not None:
                    save_dir = getattr(trainer, "default_root_dir", None) or getattr(trainer, "log_dir", None)
                if save_dir:
                    import pathlib as _pl
                    out_path = _pl.Path(save_dir) / f"init_hash_{tag}.json"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(_json.dumps({**payload, "shapes": shapes}, indent=2))
            except Exception:
                pass
        except Exception:
            pass

    def on_train_start(self) -> None:
        # Lightning may run a sanity validation before training starts; logging
        # here ensures we capture init exactly once after the model is built
        # and placed on device, regardless of sanity-val behavior.
        if not getattr(self, "_init_hash_logged_t0", False):
            self._log_init_hash("t0")
            self._init_hash_logged_t0 = True
        super_method = getattr(super(), "on_train_start", None)
        if callable(super_method):
            super_method()

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        outputs = self.forward(batch)
        if self.train_metrics or (
            self.visualize and self.trainer.global_step % self.visualize_every_n_steps == 0
        ):
            aux_outputs = self.aux_forward(batch, outputs)

        # Propagate global_step to losses that gate themselves on it
        # (e.g. VICRegFeatureLoss.ramp_start_step). Must happen BEFORE
        # compute_loss() so the loss sees the correct step.
        _step_now = int(self.trainer.global_step) if self.trainer is not None else 0
        for _loss_fn in self.loss_fns.values():
            if hasattr(_loss_fn, "_global_step"):
                _loss_fn._global_step.fill_(_step_now)

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"train/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"train/{name}": loss for name, loss in losses.items()}
            to_log["train/loss"] = total_loss

        # Phase-schedule + gate diagnostics (Round 1 fix — Critical #4).
        # We log the ramp factor (so it's visible when TubeGram/ChiBoost start
        # contributing) and each loss's confidence-gate statistics — lets us
        # verify at a glance that the gate actually fires instead of silently
        # returning 0 throughout training.
        step = int(self.trainer.global_step) if self.trainer is not None else 0
        ramp_factor = 1.0
        if self._loss_ramp_steps > 0 or self._backbone_unfreeze_step > 0:
            if step < self._backbone_unfreeze_step:
                ramp_factor = 0.0
            elif self._loss_ramp_steps > 0:
                ramp_factor = min(
                    1.0,
                    float(step - self._backbone_unfreeze_step)
                    / float(self._loss_ramp_steps),
                )
        to_log["train/tubegram/ramp"] = ramp_factor
        to_log["train/tubegram/backbone_unfrozen"] = float(self._backbone_unfrozen)
        for name, loss_fn in self.loss_fns.items():
            for attr, tag in (
                ("_last_conf_mean", "conf_mean"),
                ("_last_gate_mean", "gate_mean"),
                ("_last_gate_active_frac", "gate_active_frac"),
                # D3 MOCSP diagnostics (Round 5):
                ("_last_mask_coverage", "mocsp_mask_coverage"),
                ("_last_effective_weight", "mocsp_effective_weight"),
                # VICReg-style rank preservation (Round 9):
                ("_last_var_loss", "vicreg_var_loss"),
                ("_last_cov_loss", "vicreg_cov_loss"),
                ("_last_var_below_target_frac", "vicreg_var_below_frac"),
                ("_last_ramp_factor", "vicreg_ramp"),
            ):
                val = getattr(loss_fn, attr, None)
                if val is not None:
                    to_log[f"train/{name}/{tag}"] = val

        # Log predictor analysis metrics (if available, averaged over frames)
        if "predictor_cos_sim" in outputs["processor"]:
            to_log["train/predictor_cos_sim"] = outputs["processor"]["predictor_cos_sim"].mean()
            to_log["train/predictor_rel_change"] = outputs["processor"]["predictor_rel_change"].mean()

        # Log Hungarian match indices (fraction of identity matches)
        if "hungarian_match_indices" in outputs["processor"]:
            to_log["train/hungarian_identity_ratio"] = self._compute_identity_ratio(
                outputs["processor"]["hungarian_match_indices"],
                outputs.get("existence_masks"),
            )
        # Hungarian cost-margin diagnostic (Round 20 fix #4): runner-up minus
        # chosen cost per slot, averaged over valid (non-None) frames + slots.
        if "hungarian_cost_margin" in outputs["processor"]:
            margin_mean = self._compute_cost_margin_mean(
                outputs["processor"]["hungarian_cost_margin"]
            )
            if margin_mean is not None:
                to_log["train/hungarian_cost_margin"] = margin_mean

        # Log n_objects (for variable slot support)
        if "n_objects" in outputs:
            to_log["train/n_objects"] = outputs["n_objects"].float().mean()

        # W4 real-system diagnostics (Round 8): rate-limited feature-rank /
        # participation-ratio / target-drift / LoRA delta norm. These quantify
        # "is LoRA drifting the feature manifold" without relying on downstream
        # Hungarian identity as a proxy. Cheap (~O(D^2)) and rate-limited to
        # every 500 steps so the eigendecomp cost doesn't dominate.
        if step % 500 == 0:
            try:
                enc_bf = outputs.get("encoder", {}).get("backbone_features", None)
                if enc_bf is not None and enc_bf.numel() > 0:
                    with torch.no_grad():
                        feats = enc_bf.detach()
                        feats = feats.reshape(-1, feats.shape[-1]).float()
                        feats = feats - feats.mean(dim=0, keepdim=True)
                        cov = feats.T @ feats / max(feats.shape[0] - 1, 1)
                        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
                        total = eigvals.sum()
                        if total > 0:
                            pr = (total * total) / (eigvals.pow(2).sum() + 1e-12)
                            p = eigvals / (total + 1e-12)
                            eff_rank = torch.exp(-(p * (p + 1e-12).log()).sum())
                            to_log["train/diag/feature_pr"] = pr.item()
                            to_log["train/diag/feature_effrank"] = eff_rank.item()
                            to_log["train/diag/feature_trace"] = total.item()
                    tgt_enc = outputs.get("target_encoder", None)
                    if tgt_enc is not None and "backbone_features" in tgt_enc:
                        with torch.no_grad():
                            drift = (
                                enc_bf.detach().float()
                                - tgt_enc["backbone_features"].detach().float()
                            )
                            to_log["train/diag/target_drift_l2"] = (
                                drift.pow(2).mean().sqrt().item()
                            )
                    lora_sq = 0.0
                    lora_n = 0
                    for name, p in self.named_parameters():
                        if "lora_" in name and p.requires_grad:
                            lora_sq += p.detach().float().pow(2).sum().item()
                            lora_n += 1
                    if lora_n > 0:
                        to_log["train/diag/lora_delta_norm"] = (
                            lora_sq ** 0.5
                        )
                        to_log["train/diag/lora_n_params"] = float(lora_n)
            except Exception:
                pass

        if self.train_metrics and self.dynamics_predictor:
            prediction_batch = copy.deepcopy(batch)
            for k, v in prediction_batch.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    prediction_batch[k] = v[:, self.dynamics_predictor.history_len :]

        if self.train_metrics:
            for key, metric in self.train_metrics.items():
                if "predicted" in key.lower():
                    values = metric(**prediction_batch, **outputs, **aux_outputs)
                else:
                    values = metric(**batch, **outputs, **aux_outputs)
                self._add_metric_to_log(to_log, f"train/{key}", values)
                metric.reset()
        self.log_dict(to_log, on_step=True, on_epoch=False, batch_size=outputs["batch_size"])

        del outputs  # Explicitly delete to save memory

        if (
            self.visualize
            and self.trainer.global_step % self.visualize_every_n_steps == 0
            and self.global_rank == 0
        ):
            self._log_inputs(
                batch[self.input_key],
                {key: aux_outputs[f"{key}_hard"] for key in self.mask_keys_to_visualize},
                mode="train",
            )
            self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="train")

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if "batch_padding_mask" in batch:
            batch = self._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                return

        outputs = self.forward(batch)
        aux_outputs = self.aux_forward(batch, outputs)

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"val/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"val/{name}": loss for name, loss in losses.items()}
            to_log["val/loss"] = total_loss

        # Log predictor analysis metrics (if available, averaged over frames)
        if "predictor_cos_sim" in outputs["processor"]:
            to_log["val/predictor_cos_sim"] = outputs["processor"]["predictor_cos_sim"].mean()
            to_log["val/predictor_rel_change"] = outputs["processor"]["predictor_rel_change"].mean()

        # Log Hungarian match indices (fraction of identity matches)
        if "hungarian_match_indices" in outputs["processor"]:
            to_log["val/hungarian_identity_ratio"] = self._compute_identity_ratio(
                outputs["processor"]["hungarian_match_indices"],
                outputs.get("existence_masks"),
            )
        if "hungarian_cost_margin" in outputs["processor"]:
            margin_mean = self._compute_cost_margin_mean(
                outputs["processor"]["hungarian_cost_margin"]
            )
            if margin_mean is not None:
                to_log["val/hungarian_cost_margin"] = margin_mean

        # Log n_objects (for variable slot support)
        if "n_objects" in outputs:
            to_log["val/n_objects"] = outputs["n_objects"].float().mean()

        if self.dynamics_predictor:
            prediction_batch = deepcopy(batch)
            for k, v in prediction_batch.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    prediction_batch[k] = v[:, self.dynamics_predictor.history_len :]

        if self.val_metrics:
            for key, metric in self.val_metrics.items():
                if "predicted" in key.lower():
                    merged = {**prediction_batch, **outputs, **aux_outputs}
                else:
                    merged = {**batch, **outputs, **aux_outputs}
                metric.update(**merged)

        self.log_dict(
            to_log, on_step=False, on_epoch=True, batch_size=outputs["batch_size"], prog_bar=True
        )

        if self.visualize and batch_idx == 0 and self.global_rank == 0:
            masks_to_vis = {
                key: aux_outputs[f"{key}_vis_hard"] for key in self.mask_keys_to_visualize
            }
            if batch["segmentations"].shape[-2:] != batch[self.input_key].shape[-2:]:
                masks_to_vis["segmentations"] = self.mask_resizers["segmentation"](
                    batch["segmentations"], batch[self.input_key]
                )
            else:
                masks_to_vis["segmentations"] = batch["segmentations"]
            self._log_inputs(
                batch[self.input_key],
                masks_to_vis,
                mode="val",
            )
            self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="val")

    def validation_epoch_end(self, outputs):
        if self.val_metrics:
            to_log = {}
            for key, metric in self.val_metrics.items():
                self._add_metric_to_log(to_log, f"val/{key}", metric.compute())
                metric.reset()
            self.log_dict(to_log, prog_bar=True)

    @staticmethod
    def _add_metric_to_log(
        log_dict: Dict[str, Any], name: str, values: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ):
        if isinstance(values, dict):
            for k, v in values.items():
                log_dict[f"{name}/{k}"] = v
        else:
            log_dict[name] = values

    @staticmethod
    def _compute_identity_ratio(
        match_indices_list: List[Optional[torch.Tensor]],
        existence_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute fraction of valid slots that maintain identity mapping across frames.
        
        Args:
            match_indices_list: List of [B, N] match indices per frame (None for first frame)
            existence_masks: [B, T, N] or [B, N] mask indicating valid slots (1=valid, 0=invalid)
        """
        total_matches = 0
        identity_matches = 0
        for t, indices in enumerate(match_indices_list):
            if indices is not None:  # Skip first frame (no matching)
                B, N = indices.shape
                identity = torch.arange(N, device=indices.device).unsqueeze(0).expand(B, N)
                is_identity = (indices == identity)  # [B, N]
                
                # Only count valid slots if existence_masks provided
                if existence_masks is not None:
                    if existence_masks.dim() == 3:  # [B, T, N]
                        mask = existence_masks[:, t + 1]  # t+1 because t=0 is None (first frame)
                    else:  # [B, N] - same mask for all frames
                        mask = existence_masks
                    identity_matches += (is_identity * mask).sum().item()
                    total_matches += mask.sum().item()
                else:
                    identity_matches += is_identity.sum().item()
                    total_matches += B * N
        if total_matches == 0:
            return torch.tensor(1.0)  # No matching happened, assume identity
        return torch.tensor(identity_matches / total_matches)

    @staticmethod
    def _compute_cost_margin_mean(
        margin_list: List[Optional[torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Mean Hungarian cost margin (runner-up minus chosen) across all
        non-None per-frame [B, N] tensors. Returns None if no margins logged."""
        valid = [m for m in margin_list if m is not None]
        if not valid:
            return None
        return torch.stack([v.mean() for v in valid]).mean()

    def _log_inputs(
        self,
        inputs: torch.Tensor,
        masks_by_name: Dict[str, torch.Tensor],
        mode: str,
        step: Optional[int] = None,
    ):
        denorm = Denormalize(input_type=self.input_key)
        if step is None:
            step = self.trainer.global_step

        if self.input_key == "video":
            video = torch.stack([denorm(video) for video in inputs])
            self._log_video(f"{mode}/{self.input_key}", video, global_step=step)
            for mask_name, masks in masks_by_name.items():
                if "dynamics_predictor" in mask_name:
                    rollout_length = masks.shape[1]
                    trimmed_video = video[:, -rollout_length:]
                    video_with_masks = visualizations.mix_videos_with_masks(trimmed_video, masks)
                else:
                    video_with_masks = visualizations.mix_videos_with_masks(video, masks)
                self._log_video(
                    f"{mode}/video_with_{mask_name}",
                    video_with_masks,
                    global_step=step,
                )
        elif self.input_key == "image":
            image = denorm(inputs)
            self._log_images(f"{mode}/{self.input_key}", image, global_step=step)
            for mask_name, masks in masks_by_name.items():
                image_with_masks = visualizations.mix_images_with_masks(image, masks)
                self._log_images(
                    f"{mode}/image_with_{mask_name}",
                    image_with_masks,
                    global_step=step,
                )
        else:
            raise ValueError(f"input_type should be 'image' or 'video', but got '{self.input_key}'")

    def _log_masks(
        self,
        aux_outputs,
        mask_keys=("decoder_masks",),
        mode="val",
        types: tuple = ("frames",),
        step: Optional[int] = None,
    ):
        if step is None:
            step = self.trainer.global_step
        for mask_key in mask_keys:
            if mask_key in aux_outputs:
                masks = aux_outputs[mask_key]
                if self.input_key == "video":
                    _, f, n_obj, H, W = masks.shape
                    first_masks = masks[0].permute(1, 0, 2, 3)
                    first_masks_inverted = 1 - first_masks.reshape(n_obj, f, 1, H, W)
                    self._log_video(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                        types=types,
                    )
                elif self.input_key == "image":
                    _, n_obj, H, W = masks.shape
                    first_masks_inverted = 1 - masks[0].reshape(n_obj, 1, H, W)
                    self._log_images(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                    )
                else:
                    raise ValueError(
                        f"input_type should be 'image' or 'video', but got '{self.input_key}'"
                    )

    def _log_video(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
        max_frames: int = 8,
        types: tuple = ("frames",),
    ):
        data = data[:n_examples]
        logger = self._get_tensorboard_logger()

        if logger is not None:
            if "video" in types:
                logger.experiment.add_video(f"{name}/video", data, global_step=global_step)
            if "frames" in types:
                _, num_frames, _, _, _ = data.shape
                num_frames = min(max_frames, num_frames)
                data = data[:, :num_frames]
                data = data.flatten(0, 1)
                logger.experiment.add_image(
                    f"{name}/frames", make_grid(data, nrow=num_frames), global_step=global_step
                )

    def _save_video(self, name: str, data: torch.Tensor, global_step: int):
        assert (
            data.shape[0] == 1
        ), f"Only single videos saving are supported, but shape is: {data.shape}"
        data = data.cpu().numpy()[0].transpose(0, 2, 3, 1)
        data_dir = self.save_data_dir / name
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / f"{global_step}.npy", data)

    def _log_images(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
    ):
        n_examples = min(n_examples, data.shape[0])
        data = data[:n_examples]
        logger = self._get_tensorboard_logger()

        if logger is not None:
            logger.experiment.add_image(
                f"{name}/images", make_grid(data, nrow=n_examples), global_step=global_step
            )

    @staticmethod
    def _remove_padding(
        batch: Dict[str, Any], padding_mask: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        if torch.all(padding_mask):
            # Batch consists only of padding
            return None

        mask = ~padding_mask
        mask_as_idxs = torch.arange(len(mask))[mask.cpu()]

        output = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                output[key] = value[mask]
            elif isinstance(value, list):
                output[key] = [value[idx] for idx in mask_as_idxs]

        return output

    def _get_tensorboard_logger(self):
        if self.loggers is not None:
            for logger in self.loggers:
                if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
                    return logger
        else:
            if isinstance(self.logger, pl.loggers.tensorboard.TensorBoardLogger):
                return self.logger

    def on_load_checkpoint(self, checkpoint):
        # Reset timer during loading of the checkpoint
        # as timer is used to track time from the start
        # of the current run.
        if "callbacks" in checkpoint and "Timer" in checkpoint["callbacks"]:
            checkpoint["callbacks"]["Timer"]["time_elapsed"] = {
                "train": 0.0,
                "sanity_check": 0.0,
                "validate": 0.0,
                "test": 0.0,
                "predict": 0.0,
            }

        # Round 2 Codex Critical #4: the snapshot teacher modules
        # (``_ema_encoder``, ``_ema_initializer``, ``_ema_processor``,
        # ``_ema_decoder``) are registered as submodules at snapshot time,
        # so their weights are serialised into every Phase-2 checkpoint.
        # A fresh model has ``_ema_* is None`` (initialised in
        # ``__init__``), so ``load_state_dict`` would fail with
        # "unexpected keys" on ``_ema_*.*`` entries. Detect and
        # reconstruct the empty teacher shell BEFORE Lightning calls
        # ``load_state_dict`` so saved weights have a destination. We
        # also flip ``_backbone_unfrozen`` on — the checkpoint's teacher
        # presence implies unfreeze has already occurred.
        state_dict = checkpoint.get("state_dict", {})
        has_teacher = any(
            k.startswith(
                ("_ema_encoder.", "_ema_initializer.",
                 "_ema_processor.", "_ema_decoder.")
            )
            for k in state_dict
        )
        if has_teacher and self._ema_encoder is None:
            self._init_ema_encoder()
            self._backbone_unfrozen = True
            # LoRA params must be trainable on resume (they were at save
            # time). `_freeze_lora_params` in __init__ has already set
            # them to False for fresh runs; flip back here for resumes.
            self._unfreeze_lora_params()

    def load_weights_from_checkpoint(
        self, checkpoint_path: str, module_mapping: Optional[Dict[str, str]] = None
    ):
        """Load weights from a checkpoint into the specified modules."""
        checkpoint = torch.load(checkpoint_path)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        if module_mapping is None:
            module_mapping = {
                key.split(".")[0]: key.split(".")[0]
                for key in checkpoint
                if hasattr(self, key.split(".")[0])
            }

        for dest_module, source_module in module_mapping.items():
            try:
                module = utils.read_path(self, dest_module)
            except ValueError:
                raise ValueError(f"Module {dest_module} could not be retrieved from model") from None

            state_dict = {}
            for key, weights in checkpoint.items():
                if key.startswith(source_module):
                    if key != source_module:
                        key = key[len(source_module + ".") :]  # Remove prefix
                    state_dict[key] = weights
            if len(state_dict) == 0:
                raise ValueError(
                    f"No weights for module {source_module} found in checkpoint {checkpoint_path}."
                )

            module.load_state_dict(state_dict)
