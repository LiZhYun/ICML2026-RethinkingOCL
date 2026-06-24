from typing import Any, Dict, List, Optional, Union

import einops
import timm
import torch
import torchvision
from torch import nn

from slotcontrast.modules import utils
from slotcontrast.utils import config_as_kwargs, make_build_fn


@make_build_fn(__name__, "encoder")
def build(config, name: str):
    if name == "FrameEncoder":
        pos_embed = None
        if config.get("pos_embed") and config.get("use_pos_embed"):
            pos_embed = utils.build_module(config.pos_embed)

        output_transform = None
        if config.get("output_transform"):
            output_transform = utils.build_module(config.output_transform)
        return FrameEncoder(
            backbone=utils.build_module(config.backbone, default_group="encoders"),
            pos_embed=pos_embed,
            output_transform=output_transform,
            main_features_key=config.get("main_features_key", "vit_block12"),
            **config_as_kwargs(config, ("backbone", "pos_embed", "output_transform", "main_features_key")),
        )
    else:
        return None


class FrameEncoder(nn.Module):
    """Module reducing image to set of features."""

    def __init__(
        self,
        backbone: nn.Module,
        pos_embed: Optional[nn.Module] = None,
        output_transform: Optional[nn.Module] = None,
        spatial_flatten: bool = False,
        main_features_key: str = "vit_block12",
        normalize_features: bool = False,  # Add this parameter
        normalization_type: str = "l2",    # 'l2' or 'standardize'
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.pos_embed = pos_embed
        self.output_transform = output_transform
        self.spatial_flatten = spatial_flatten
        self.main_features_key = main_features_key
        self.normalize_features = normalize_features
        self.normalization_type = normalization_type

    def forward(
        self, images: torch.Tensor, camera_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # images: batch x n_channels x height x width
        # camera_data: optional dict with 'depth', 'intrinsics', 'extrinsics'
        backbone_features = self.backbone(images)
        if isinstance(backbone_features, dict):
            features = backbone_features[self.main_features_key].clone()
        else:
            features = backbone_features.clone()

        # ADD NORMALIZATION HERE - before positional embeddings
        if self.normalize_features:
            if self.normalization_type == "l2":
                # L2 normalization (for retrieval/similarity tasks)
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
            elif self.normalization_type == "standardize":
                # Standardization (for classification/dense tasks)
                features = (features - features.mean()) / (features.std() + 1e-6)

        if self.pos_embed:
            # Prepare kwargs for pos_embed (e.g., camera_data for 3D embeddings)
            pos_kwargs = camera_data if camera_data is not None else {}
            
            # Handle shape conversion for spatial (4D) positional embeddings
            needs_reshape = False
            if features.ndim == 3:
                # Check if pos_embed expects 4D input
                expects_4d = (
                    isinstance(self.pos_embed, utils.CoordinatePositionEmbed) or
                    (hasattr(self.pos_embed, 'expected_dims') and self.pos_embed.expected_dims == 4)
                )
                if expects_4d:
                    needs_reshape = True
                    B, N, D = features.shape
                    H = W = int(N ** 0.5)
                    assert H * W == N, f"Cannot reshape {N} patches to square grid"
                    features = features.transpose(1, 2).view(B, D, H, W)
            
            features = self.pos_embed(features, **pos_kwargs)
            
            # Reshape back to 3D if needed
            if needs_reshape:
                features = features.flatten(2).transpose(1, 2)
            
            backbone_features = features.clone()

        if self.spatial_flatten:
            features = einops.rearrange(features, "b c h w -> b (h w) c")
        if self.output_transform:
            features = self.output_transform(features)

        assert (
            features.ndim == 3
        ), f"Expect output shape (batch, tokens, dims), but got {features.shape}"
        if isinstance(backbone_features, dict):
            for k, backbone_feature in backbone_features.items():
                if self.spatial_flatten:
                    backbone_features[k] = einops.rearrange(backbone_feature, "b c h w -> b (h w) c")
                assert (
                    backbone_feature.ndim == 3
                ), f"Expect output shape (batch, tokens, dims), but got {backbone_feature.shape}"
            main_backbone_features = backbone_features[self.main_features_key]

            return {
                "features": features,
                "backbone_features": main_backbone_features,
                **backbone_features,
            }
        else:
            if self.spatial_flatten:
                backbone_features = einops.rearrange(backbone_features, "b c h w -> b (h w) c")
            assert (
                backbone_features.ndim == 3
            ), f"Expect output shape (batch, tokens, dims), but got {backbone_features.shape}"

            return {
                "features": features,
                "backbone_features": backbone_features,
            }


class TimmExtractor(nn.Module):
    """Feature extractor utilizing models from timm library.
    
    Supports ViT models with different depths:
    - ViT-Small/Base: 12 blocks (use vit_block1-12)
    - ViT-Large: 24 blocks (use vit_block1-24)
    - ViT-Huge: 32 blocks (use vit_block1-32)
    """

    # Convenience aliases for feature keys (supports up to 32 blocks for ViT-Huge)
    FEATURE_ALIASES = {
        **{f"resnet_block{i}": f"layer{i}" for i in range(1, 5)},
        **{f"vit_block{i + 1}": f"blocks.{i}" for i in range(32)},
        **{f"vit_block_values{i + 1}": f"blocks.{i}.attn.qkv" for i in range(32)},
        **{f"vit_block_queries{i + 1}": f"blocks.{i}.attn.qkv" for i in range(32)},
        **{f"vit_block_keys{i + 1}": f"blocks.{i}.attn.qkv" for i in range(32)},
        "vit_output": "norm",
    }
    FEATURE_MAPPING = {
        **{f"layer{i}": f"resnet_block{i}" for i in range(1, 5)},
        **{f"blocks.{i}": f"vit_block{i + 1}" for i in range(32)},
        **{f"blocks.{i}.attn.qkv": f"vit_block_keys{i + 1}" for i in range(32)},
        "norm": "vit_output",
    }

    def __init__(
        self,
        model: str,
        pretrained: bool = False,
        frozen: bool = False,
        features: Optional[Union[str, List[str]]] = None,
        checkpoint_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        adapt_mode: Optional[str] = None,
        adapt_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        model_name = model
        self.frozen = frozen
        self.features = [features] if isinstance(features, str) else features
        self.is_vit = model_name.startswith("vit")
        self.has_lora = lora_config is not None
        # Parameter-efficient adaptation alternatives to LoRA, for the
        # TPAMI rev's "is LoRA special vs just parameter-efficient?" check.
        # adapt_mode ∈ {None, "bitfit", "last_k_blocks"}; adapt_config is
        # mode-specific (e.g. {"k": 2} for last_k_blocks).
        self.adapt_mode = adapt_mode
        self.adapt_config = adapt_config or {}

        model = TimmExtractor._create_model(model_name, pretrained, checkpoint_path, model_kwargs)

        if lora_config is not None:
            from slotcontrast.modules.lora import apply_lora, lora_param_count
            n = apply_lora(
                model,
                r=lora_config.get("r", 8),
                alpha=lora_config.get("alpha", 16.0),
                target_modules=lora_config.get("target_modules", None),
            )
            info = lora_param_count(model)
            print(f"[LoRA] Adapted {n} layers | trainable: {info['trainable']:,} "
                  f"/ total: {info['total']:,} ({100*info['trainable']/info['total']:.2f}%)")

        # Use hooks instead of create_feature_extractor to avoid FX GraphModule issues
        self.feature_outputs = {}
        self.hooks = []

        if self.features is not None:
            # Register forward hooks to capture intermediate features
            def get_hook(name):
                def hook(module, input, output):
                    self.feature_outputs[name] = output
                return hook

            for feature_name in self.features:
                # Translate aliases
                target_name = feature_name
                if feature_name in TimmExtractor.FEATURE_ALIASES:
                    target_name = TimmExtractor.FEATURE_ALIASES[feature_name]

                # Navigate to the target module and register hook
                parts = target_name.split('.')
                target_module = model
                for part in parts:
                    if part.isdigit():
                        target_module = target_module[int(part)]
                    else:
                        target_module = getattr(target_module, part)

                handle = target_module.register_forward_hook(get_hook(self.FEATURE_MAPPING.get(target_name, feature_name)))
                self.hooks.append(handle)

        self.model = model

        if self.frozen and not self.has_lora:
            self.requires_grad_(False)

        # Apply parameter-efficient adaptation modes AFTER any frozen() call.
        # Both modes assume frozen=true upstream and selectively un-freeze
        # specific parameter subsets.
        if self.adapt_mode == "bitfit":
            # BitFit (Zaken et al., 2022): tune biases only.
            n_total = 0
            n_trainable = 0
            for pname, p in self.model.named_parameters():
                n_total += p.numel()
                is_bias = pname.endswith(".bias") or pname.endswith("_bias")
                p.requires_grad_(is_bias)
                if is_bias:
                    n_trainable += p.numel()
            print(f"[BitFit] biases-only | trainable: {n_trainable:,} / "
                  f"total: {n_total:,} ({100*n_trainable/n_total:.3f}%)")
        elif self.adapt_mode == "last_k_blocks":
            k = int(self.adapt_config.get("k", 2))
            # Find the transformer blocks list (timm ViT exposes `.blocks`).
            blocks = getattr(self.model, "blocks", None)
            if blocks is None:
                raise ValueError(
                    "adapt_mode='last_k_blocks' requires a timm ViT with .blocks; "
                    f"got model={model_name}"
                )
            n_blocks = len(blocks)
            unfreeze_from = max(0, n_blocks - k)
            # Freeze everything, then unfreeze the last-k blocks + the final norm.
            self.model.requires_grad_(False)
            for i in range(unfreeze_from, n_blocks):
                blocks[i].requires_grad_(True)
            # Also unfreeze the final LN if it exists (`norm` for ViT).
            final_norm = getattr(self.model, "norm", None)
            if final_norm is not None:
                final_norm.requires_grad_(True)
            n_total = sum(p.numel() for p in self.model.parameters())
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[LastK] k={k} of {n_blocks} blocks unfrozen "
                  f"(from index {unfreeze_from}) | trainable: {n_trainable:,} "
                  f"/ total: {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    @staticmethod
    def _create_model(
        model_name: str,
        pretrained: bool,
        checkpoint_path: Optional[str],
        model_kwargs: Optional[Dict[str, Any]],
        trials: int = 0,
    ) -> nn.Module:
        if model_kwargs is None:
            model_kwargs = {}

        try:
            model = timm.create_model(
                model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, **model_kwargs
            )
        except (FileExistsError, FileNotFoundError):
            # Timm uses Hugginface hub for loading the files, which does some symlinking in the
            # background when loading the checkpoint. When multiple concurrent jobs attempt to
            # load the checkpoint, this can create conflicts, because the symlink is first removed,
            # then created again by each job. We attempt to catch the resulting errors here, and
            # retry creating the model, up to 3 times.
            if trials == 2:
                raise
            else:
                model = None

        if model is None:
            model = TimmExtractor._create_model(
                model_name, pretrained, checkpoint_path, model_kwargs, trials=trials + 1
            )

        return model

    def forward(self, inp):
        # Clear previous feature outputs
        self.feature_outputs.clear()
        
        if self.frozen:
            with torch.no_grad():
                outputs = self.model(inp)
        else:
            outputs = self.model(inp)

        if self.features is not None:
            # Use captured features from hooks
            outputs = self.feature_outputs
            
            if self.is_vit:
                # Remove CLS token and register tokens
                # DINOv2: [CLS, patch1, patch2, ...] -> remove 1 token
                # DINOv3: [CLS, reg1, reg2, reg3, reg4, patch1, ...] -> remove 5 tokens
                # Check if model has register tokens (DINOv3)
                n_prefix_tokens = getattr(self.model, 'num_prefix_tokens', 1)
                outputs = {k: v[:, n_prefix_tokens:] for k, v in outputs.items()}
            
            for name in list(outputs.keys()):
                if ("keys" in name) or ("queries" in name) or ("values" in name):
                    feature_name = name.replace("queries", "keys").replace("values", "keys")
                    B, N, C = outputs[feature_name].shape
                    qkv = outputs[feature_name].reshape(
                        B, N, 3, C // 3
                    )  # outp has shape B, N, 3 * H * (C // H)
                    q, k, v = qkv.unbind(2)
                    if "keys" in name:
                        outputs[name] = k
                    elif "queries" in name:
                        outputs[name] = q
                    elif "values" in name:
                        outputs[name] = v
                    else:
                        raise ValueError(f"Unknown feature name {name}.")

            if len(outputs) == 1:
                # Unpack single output for now
                return next(iter(outputs.values()))
            else:
                return outputs
        else:
            return outputs


# --- MOCSP (D3) encoder extensions --------------------------------------
# Re-export MaskingFrameEncoder so the encoder build registry resolves it
# via ``get_class_by_name``. The subclass itself lives in a separate file
# so the base ``FrameEncoder`` is untouched.
from slotcontrast.modules.encoders_mocsp import MaskingFrameEncoder  # noqa: E402, F401
