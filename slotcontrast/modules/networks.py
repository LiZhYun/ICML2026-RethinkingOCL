import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import einops
import torch
from torch import nn
from torch.nn import functional as F

from slotcontrast.modules import utils
from slotcontrast.utils import make_build_fn

# Default weight init for MLP, CNNEncoder, CNNDecoder
DEFAULT_WEIGHT_INIT = "default"


@make_build_fn(__name__, "network module")
def build(config, name: str):
    if name == "two_layer_mlp":
        inp_dim = None
        outp_dim = None
        frozen = False
        if "dim" in config:
            inp_dim = config["dim"]
            outp_dim = config["dim"]
        if "inp_dim" in config:
            inp_dim = config["inp_dim"]
        if "outp_dim" in config:
            outp_dim = config["outp_dim"]
        if "outp_dim" in config:
            outp_dim = config["outp_dim"]

        if inp_dim is None:
            raise ValueError("Specify input dimensions with `inp_dim` or `dim`")
        if outp_dim is None:
            raise ValueError("Specify output dimension with `outp_dim` or `dim`")

        hidden_dims = [config.get("hidden_dim", 4 * inp_dim)]
        layer_norm = config.get("layer_norm") or config.get("initial_layer_norm", False)
        residual = config.get("residual", False)
        activation = config.get("activation", "relu")
        final_activation = config.get("final_activation", False)
        weight_init = config.get("weight_init", DEFAULT_WEIGHT_INIT)

        return MLP(
            inp_dim,
            outp_dim,
            hidden_dims,
            layer_norm,
            activation,
            final_activation,
            residual,
            weight_init,
            frozen,
        )
    elif name == "slot_attention_encoder" or name.startswith("savi_cnn_encoder"):
        inp_dim = config.get("inp_dim", 3)

        if name == "slot_attention_encoder":
            feature_multiplier = 1
            downsamplings = 0
        elif name == "savi_cnn_encoder":
            feature_multiplier = 1
            downsamplings = 1
        elif name == "savi_cnn_encoder_64":
            feature_multiplier = 0.5
            downsamplings = 0

        feature_multiplier = config.get("feature_multiplier", feature_multiplier)
        downsamplings = config.get("downsamplings", downsamplings)
        weight_init = config.get("weight_init", DEFAULT_WEIGHT_INIT)

        return make_slot_attention_encoder(inp_dim, feature_multiplier, downsamplings, weight_init)
    elif name.startswith("savi_decoder"):
        inp_dim = config.get("inp_dim")
        if inp_dim is None:
            raise ValueError("Need to specify input dimensions with `inp_dim`")

        if name == "savi_decoder":
            upsamplings = 4
        elif name == "savi_decoder_64":
            upsamplings = 3

        upsamplings = config.get("upsamplings", upsamplings)
        weight_init = config.get("weight_init", DEFAULT_WEIGHT_INIT)

        return make_savi_decoder(
            inp_dim, config.get("feature_multiplier", 1), upsamplings, weight_init
        )
    else:
        return None


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims: List[int],
        initial_layer_norm: bool = False,
        activation: Union[str, nn.Module] = "relu",
        final_activation: Union[bool, str] = False,
        residual: bool = False,
        weight_init: str = DEFAULT_WEIGHT_INIT,
        frozen: bool = False,
    ):
        super().__init__()
        self.residual = residual
        if residual:
            assert inp_dim == outp_dim

        layers = []
        if initial_layer_norm:
            layers.append(nn.LayerNorm(inp_dim))

        cur_dim = inp_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(cur_dim, dim))
            layers.append(utils.get_activation_fn(activation))
            cur_dim = dim

        layers.append(nn.Linear(cur_dim, outp_dim))
        if final_activation:
            if isinstance(final_activation, bool):
                final_activation = "relu"
            layers.append(utils.get_activation_fn(final_activation))

        self.layers = nn.Sequential(*layers)
        utils.init_parameters(self.layers, weight_init)

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        outp = self.layers(inp)

        if self.residual:
            return inp + outp
        else:
            return outp


def _infer_common_length(fail_on_missing_length=True, **kwargs) -> int:
    """Given kwargs of scalars and lists, checks that all lists have the same length and returns it.

    Optionally fails if no length was provided.
    """
    length = None
    name = None
    for cur_name, arg in kwargs.items():
        if isinstance(arg, (tuple, list)):
            cur_length = len(arg)
            if length is None:
                length = cur_length
                name = cur_name
            elif cur_length != length:
                raise ValueError(
                    f"Inconsistent lengths: {cur_name} has length {cur_length}, "
                    f"but {name} has length {length}"
                )

    if fail_on_missing_length and length is None:
        names = ", ".join(f"`{key}`" for key in kwargs.keys())
        raise ValueError(f"Need to specify a list for at least one of {names}.")

    return length


def _maybe_expand_list(arg: Union[int, List[int]], length: int) -> list:
    if not isinstance(arg, (tuple, list)):
        return [arg] * length

    return list(arg)


class CNNEncoder(nn.Sequential):
    """Simple convolutional encoder.

    For `features`, `kernel_sizes`, `strides`, scalars can be used to avoid repeating arguments,
    but at least one list needs to be provided to specify the number of layers.
    """

    def __init__(
        self,
        inp_dim: int,
        features: Union[int, List[int]],
        kernel_sizes: Union[int, List[int]],
        strides: Union[int, List[int]] = 1,
        outp_dim: Optional[int] = None,
        weight_init: str = "default",
    ):
        length = _infer_common_length(features=features, kernel_sizes=kernel_sizes, strides=strides)
        features = _maybe_expand_list(features, length)
        kernel_sizes = _maybe_expand_list(kernel_sizes, length)
        strides = _maybe_expand_list(strides, length)

        layers = []
        cur_dim = inp_dim
        for dim, kernel_size, stride in zip(features, kernel_sizes, strides):
            layers.append(
                nn.Conv2d(
                    cur_dim,
                    dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=self.get_same_padding(kernel_size, stride),
                )
            )
            layers.append(nn.ReLU(inplace=True))
            cur_dim = dim

        if outp_dim is not None:
            layers.append(nn.Conv1d(cur_dim, outp_dim, kernel_size=1, stride=1))

        super().__init__(*layers)
        utils.init_parameters(self, weight_init)

    @staticmethod
    def get_same_padding(kernel_size: int, stride: int) -> Union[str, int]:
        """Try to infer same padding for convolutions."""
        # This method is very lazily implemented, but oh well..
        if stride == 1:
            return "same"
        if kernel_size == 3:
            if stride == 2:
                return 1
        elif kernel_size == 5:
            if stride == 2:
                return 2

        raise ValueError(f"Don't know 'same' padding for kernel {kernel_size}, stride {stride}")


def make_slot_attention_encoder(
    inp_dim: int,
    feature_multiplier: float = 1,
    downsamplings: int = 0,
    weight_init: str = DEFAULT_WEIGHT_INIT,
) -> CNNEncoder:
    """CNN encoder as used in Slot Attention paper.

    By default, 4 layers with 64 channels each, keeping the spatial input resolution the same.

    This encoder is also used by SAVi, in the following configurations:

    - for image resolution 64: feature_multiplier=0.5, downsamplings=0
    - for image resolution 128: feature_multiplier=1, downsamplings=1

    and STEVE, in the following configurations:

    - for image resolution 64: feature_multiplier=1, downsamplings=0
    - for image resolution 128: feature_multiplier=1, downsamplings=1
    """
    assert 0 <= downsamplings <= 4
    channels = int(64 * feature_multiplier)
    strides = [2] * downsamplings + [1] * (4 - downsamplings)
    return CNNEncoder(
        inp_dim,
        features=[channels, channels, channels, channels],
        kernel_sizes=[5, 5, 5, 5],
        strides=strides,
        weight_init=weight_init,
    )


class CNNDecoder(nn.Sequential):
    """Simple convolutional decoder.

    For `features`, `kernel_sizes`, `strides`, scalars can be used to avoid repeating arguments,
    but at least one list needs to be provided to specify the number of layers.
    """

    def __init__(
        self,
        inp_dim: int,
        features: Union[int, List[int]],
        kernel_sizes: Union[int, List[int]],
        strides: Union[int, List[int]] = 1,
        outp_dim: Optional[int] = None,
        weight_init: str = DEFAULT_WEIGHT_INIT,
    ):
        length = _infer_common_length(features=features, kernel_sizes=kernel_sizes, strides=strides)
        features = _maybe_expand_list(features, length)
        kernel_sizes = _maybe_expand_list(kernel_sizes, length)
        strides = _maybe_expand_list(strides, length)

        layers = []
        cur_dim = inp_dim
        for dim, kernel_size, stride in zip(features, kernel_sizes, strides):
            padding, output_padding = self.get_same_padding(kernel_size, stride)
            layers.append(
                nn.ConvTranspose2d(
                    cur_dim,
                    dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            cur_dim = dim

        if outp_dim is not None:
            layers.append(nn.Conv1d(cur_dim, outp_dim, kernel_size=1, stride=1))

        super().__init__(*layers)
        utils.init_parameters(self, weight_init)

    @staticmethod
    def get_same_padding(kernel_size: int, stride: int) -> Tuple[int, int]:
        """Try to infer same padding for transposed convolutions."""
        # This method is very lazily implemented, but oh well..
        if kernel_size == 3:
            if stride == 1:
                return 1, 0
            if stride == 2:
                return 1, 1
        elif kernel_size == 5:
            if stride == 1:
                return 2, 0
            if stride == 2:
                return 2, 1

        raise ValueError(f"Don't know 'same' padding for kernel {kernel_size}, stride {stride}")


def make_savi_decoder(
    inp_dim: int,
    feature_multiplier: float = 1,
    upsamplings: int = 4,
    weight_init: str = DEFAULT_WEIGHT_INIT,
) -> CNNDecoder:
    """CNN encoder as used in SAVi paper.

    By default, 4 layers with 64 channels each, upscaling from a 8x8 feature map to 128x128.
    """
    assert 0 <= upsamplings <= 4
    channels = int(64 * feature_multiplier)
    strides = [2] * upsamplings + [1] * (4 - upsamplings)
    return CNNDecoder(
        inp_dim,
        features=[channels, channels, channels, channels],
        kernel_sizes=[5, 5, 5, 5],
        strides=strides,
        weight_init=weight_init,
    )


class Attention(nn.Module):
    """Multihead attention.

    Adapted from timm's ViT implementation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        inner_dim: Optional[int] = None,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        kdim = dim if kdim is None else kdim
        vdim = dim if vdim is None else vdim
        inner_dim = dim if inner_dim is None else inner_dim
        if inner_dim % num_heads != 0:
            raise ValueError("`inner_dim` must be divisible by `num_heads`")

        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.head_dim = inner_dim // num_heads
        self.scale = self.head_dim**-0.5

        self._same_qkv_dim = dim == kdim and dim == vdim
        self._same_kv_dim = kdim == vdim

        if self._same_qkv_dim:
            self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        elif self._same_kv_dim:
            self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
            self.kv = nn.Linear(kdim, inner_dim * 2, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
            self.k = nn.Linear(kdim, inner_dim, bias=qkv_bias)
            self.v = nn.Linear(vdim, inner_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.out_proj_drop = nn.Dropout(proj_drop)

        self.init_parameters()

    def init_parameters(self):
        if self._same_qkv_dim:
            bound = math.sqrt(6.0 / (self.qkv.weight.shape[0] // 3 + self.qkv.weight.shape[1]))
            nn.init.uniform_(self.qkv.weight, -bound, bound)  # Xavier init for separate Q, K, V
            if self.qkv.bias is not None:
                nn.init.constant_(self.qkv.bias, 0.0)
        elif self._same_kv_dim:
            utils.init_parameters(self.q, "xavier_uniform")
            bound = math.sqrt(6.0 / (self.kv.weight.shape[0] // 2 + self.kv.weight.shape[1]))
            nn.init.uniform_(self.kv.weight, -bound, bound)  # Xavier init for separate K, V
            if self.kv.bias is not None:
                nn.init.constant_(self.kv.bias, 0.0)
        else:
            utils.init_parameters((self.q, self.k, self.v), "xavier_uniform")

        utils.init_parameters(self.out_proj, "xavier_uniform")

    def _in_proj(self, q, k, v):
        """Efficiently compute in-projection.

        Adapted from torch.nn.functional.multi_head_attention.
        """
        if self._same_qkv_dim:
            w_kv = b_kv = b_q = b_k = b_v = None
            w = self.qkv.weight
            b = self.qkv.bias if hasattr(self.qkv, "bias") else None
        elif self._same_kv_dim:
            w = b = b_k = b_v = None
            w_q = self.q.weight
            w_kv = self.kv.weight
            b_q = self.q.bias if hasattr(self.q, "bias") else None
            b_kv = self.kv.bias if hasattr(self.kv, "bias") else None
        else:
            w = w_kv = b = b_kv = None
            w_q = self.q.weight
            w_k = self.k.weight
            w_v = self.v.weight
            b_q = self.q.bias if hasattr(self.q, "bias") else None
            b_k = self.k.bias if hasattr(self.k, "bias") else None
            b_v = self.v.bias if hasattr(self.v, "bias") else None

        if k is v:
            if q is k:
                # Self-attention
                return F.linear(q, w, b).chunk(3, dim=-1)
            else:
                # Encoder-decoder attention
                if w is not None:
                    dim = w.shape[0] // 3
                    w_q, w_kv = w.split([dim, dim * 2])
                    if b is not None:
                        b_q, b_kv = b.split([dim, dim * 2])
                return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
        else:
            if w is not None:
                w_q, w_k, w_v = w.chunk(3)
                if b is not None:
                    b_q, b_k, b_v = b.chunk(3)
            elif w_kv is not None:
                w_k, w_v = w_kv.chunk(2)
                if b_kv is not None:
                    b_k, b_v = b_kv.chunk(2)

            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = key if key is not None else query
        value = value if value is not None else query

        bs, n_queries, _ = query.shape
        n_keys = key.shape[1]

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                expected = (n_queries, n_keys)
                if attn_mask.shape != expected:
                    raise ValueError(
                        f"2D `attn_mask` should have shape {expected}, but has "
                        f"shape {attn_mask.shape}"
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.ndim == 3:
                expected = (bs * self.num_heads, n_queries, n_keys)
                if attn_mask.shape != expected:
                    raise ValueError(
                        f"3D `attn_mask` should have shape {expected}, but has "
                        f"shape {attn_mask.shape}"
                    )
        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool
            expected = (bs, n_keys)
            if key_padding_mask.shape != expected:
                raise ValueError(
                    f"`key_padding_mask` should have shape {expected}, but has shape "
                    f"{key_padding_mask.shape}"
                )
            key_padding_mask = einops.repeat(
                key_padding_mask, "b n -> (b h) 1 n", b=bs, h=self.num_heads, n=n_keys
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        q, k, v = self._in_proj(query, key, value)

        q = einops.rearrange(q, "b n (h d) -> (b h) n d", h=self.num_heads, d=self.head_dim)
        k = einops.rearrange(k, "b n (h d) -> (b h) n d", h=self.num_heads, d=self.head_dim)
        v = einops.rearrange(v, "b n (h d) -> (b h) n d", h=self.num_heads, d=self.head_dim)

        q_scaled = q / self.scale
        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn = torch.bmm(q_scaled, k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)  # (B x H) x N x M
        pre_dropout_attn = attn
        attn = self.attn_drop(attn)

        weighted_v = attn @ v
        x = einops.rearrange(weighted_v, "(b h) n d -> b n (h d)", h=self.num_heads, d=self.head_dim)
        x = self.out_proj(x)
        x = self.out_proj_drop(x)

        if return_weights:
            weights = einops.rearrange(pre_dropout_attn, "(b h) n m -> b h n m", h=self.num_heads)
            return x, weights.mean(dim=1)
        else:
            return x, None


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Like torch.nn.TransformerEncoderLayer, but with customizations."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dim_attn: Optional[int] = None,
        dim_kv: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = torch.nn.functional.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        initial_residual_scale: Optional[float] = None,
        use_gated: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device=device,
            dtype=dtype,
        )
        self.self_attn = Attention(
            dim=d_model,
            num_heads=nhead,
            kdim=dim_kv,
            vdim=dim_kv,
            inner_dim=dim_attn,
            qkv_bias=qkv_bias,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        self.use_gated = use_gated
        if use_gated:
            self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        if initial_residual_scale is not None:
            self.scale1 = utils.LayerScale(d_model, init_values=initial_residual_scale)
            self.scale2 = utils.LayerScale(d_model, init_values=initial_residual_scale)
        else:
            self.scale1 = nn.Identity()
            self.scale2 = nn.Identity()

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        keys = keys if keys is not None else x
        values = values if values is not None else x
        x, attn = self.self_attn(
            x,
            keys,
            values,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_weights=return_weights,
        )
        x = self.dropout1(x)

        if return_weights:
            return x, attn
        else:
            return x

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        x = src
        attn = None
        if self.norm_first:
            if return_weights:
                residual, attn = self._sa_block(
                    self.norm1(x), src_mask, src_key_padding_mask, keys=memory, values=memory, return_weights=True
                )
            else:
                residual = self._sa_block(
                    self.norm1(x), src_mask, src_key_padding_mask, keys=memory, values=memory
                )
            if self.use_gated:
                gate = torch.sigmoid(self.gate_proj(x))
                residual = residual * gate
            x = x + self.scale1(residual)
            x = x + self.scale2(self._ff_block(self.norm2(x)))
        else:
            if return_weights:
                residual, attn = self._sa_block(x, src_mask, src_key_padding_mask, keys=memory, values=memory, return_weights=True)
            else:
                residual = self._sa_block(x, src_mask, src_key_padding_mask, keys=memory, values=memory)
            if self.use_gated:
                gate = torch.sigmoid(self.gate_proj(x))
                residual = residual * gate
            x = self.norm1(x + self.scale1(residual))
            x = self.norm2(x + self.scale2(self._ff_block(x)))

        if return_weights:
            return x, attn
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_blocks: int,
        n_heads: int,
        qkv_dim: Optional[int] = None,
        memory_dim: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "relu",
        hidden_dim: Optional[int] = None,
        initial_residual_scale: Optional[float] = None,
        use_gated: bool = False,
        frozen: bool = False,
        **kwargs,  # Absorb extra config arguments for compatibility
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim,
                    n_heads,
                    dim_feedforward=hidden_dim,
                    dim_attn=qkv_dim,
                    dim_kv=memory_dim,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=1e-05,
                    batch_first=True,
                    norm_first=True,
                    initial_residual_scale=initial_residual_scale,
                    use_gated=use_gated,
                )
                for _ in range(n_blocks)
            ]
        )

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def forward(
        self,
        inp: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        x = inp
        attn_list = [] if return_weights else None

        for block in self.blocks:
            if return_weights:
                x, attn = block(x, mask, key_padding_mask, memory, return_weights=True)
                attn_list.append(attn)
            else:
                x = block(x, mask, key_padding_mask, memory)

        if return_weights:
            return x, attn_list
        return x


class CrossAttentionEncoderLayer(TransformerEncoderLayer):
    """TransformerEncoderLayer with additional cross-attention block."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dim_attn: Optional[int] = None,
        dim_kv: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = torch.nn.functional.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        initial_residual_scale: Optional[float] = None,
        use_gated: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model, nhead, dim_feedforward, dim_attn, dim_kv, qkv_bias,
            dropout, activation, layer_norm_eps, batch_first, norm_first,
            initial_residual_scale, use_gated, device, dtype,
        )
        # Cross-attention components
        self.cross_attn = Attention(
            dim=d_model,
            num_heads=nhead,
            kdim=dim_kv,
            vdim=dim_kv,
            inner_dim=dim_attn,
            qkv_bias=qkv_bias,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.norm_ca = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout_ca = nn.Dropout(dropout)
        if initial_residual_scale is not None:
            self.scale_ca = utils.LayerScale(d_model, init_values=initial_residual_scale)
        else:
            self.scale_ca = nn.Identity()

    def _ca_block(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, _ = self.cross_attn(x, memory, memory, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout_ca(x)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        cross_memory: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        x = src
        attn = None
        if self.norm_first:
            # Cross-attention first (if cross_memory provided) - decoder-style
            if cross_memory is not None:
                x = x + self.scale_ca(self._ca_block(self.norm_ca(x), cross_memory))
            # Self-attention
            if return_weights:
                residual, attn = self._sa_block(
                    self.norm1(x), src_mask, src_key_padding_mask, keys=memory, values=memory, return_weights=True
                )
            else:
                residual = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, keys=memory, values=memory)
            if self.use_gated:
                gate = torch.sigmoid(self.gate_proj(x))
                residual = residual * gate
            x = x + self.scale1(residual)
            # FFN
            x = x + self.scale2(self._ff_block(self.norm2(x)))
        else:
            # Cross-attention first (if cross_memory provided) - decoder-style
            if cross_memory is not None:
                x = self.norm_ca(x + self.scale_ca(self._ca_block(x, cross_memory)))
            # Self-attention
            if return_weights:
                residual, attn = self._sa_block(x, src_mask, src_key_padding_mask, keys=memory, values=memory, return_weights=True)
            else:
                residual = self._sa_block(x, src_mask, src_key_padding_mask, keys=memory, values=memory)
            if self.use_gated:
                gate = torch.sigmoid(self.gate_proj(x))
                residual = residual * gate
            x = self.norm1(x + self.scale1(residual))
            # FFN
            x = self.norm2(x + self.scale2(self._ff_block(x)))

        if return_weights:
            return x, attn
        return x


class CrossAttentionPredictor(nn.Module):
    """Predictor with cross-attention to per-frame initialized slots.
    
    Same structure as TransformerEncoder but uses CrossAttentionEncoderLayer.
    """

    def __init__(
        self,
        dim: int,
        n_blocks: int = 2,
        n_heads: int = 4,
        qkv_dim: Optional[int] = None,
        memory_dim: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "relu",
        hidden_dim: Optional[int] = None,
        initial_residual_scale: Optional[float] = None,
        use_gated: bool = False,
        frozen: bool = False,
        **kwargs,  # Absorb extra config arguments for compatibility
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.blocks = nn.ModuleList([
            CrossAttentionEncoderLayer(
                dim, n_heads,
                dim_feedforward=hidden_dim,
                dim_attn=qkv_dim,
                dim_kv=memory_dim,
                qkv_bias=qkv_bias,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=1e-05,
                batch_first=True,
                norm_first=True,
                initial_residual_scale=initial_residual_scale,
                use_gated=use_gated,
            )
            for _ in range(n_blocks)
        ])

        # For detection in LatentProcessor
        self.cross_attn = True

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def forward(
        self,
        inp: torch.Tensor,
        init_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        x = inp
        attn_list = [] if return_weights else None

        for block in self.blocks:
            if return_weights:
                x, attn = block(x, mask, key_padding_mask, memory, cross_memory=init_state, return_weights=True)
                attn_list.append(attn)
            else:
                x = block(x, mask, key_padding_mask, memory, cross_memory=init_state)

        if return_weights:
            return x, attn_list
        return x


class MemoryConditionedLayer(nn.Module):
    """Single layer with self-attention and optional cross-attention to memory."""

    def __init__(
        self,
        dim: int,
        memory_dim: int,
        n_heads: int,
        hidden_dim: int,
        dropout: float,
        activation: str,
        use_memory: bool = True,
        use_gated: bool = False,
    ):
        super().__init__()
        self.use_memory = use_memory
        self.use_gated = use_gated

        # Self-attention (always present)
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)

        if use_gated:
            self.gate_proj_self = nn.Linear(dim, dim, bias=True)

        # Cross-attention to memory (conditional)
        if use_memory:
            self.cross_attn = nn.MultiheadAttention(
                dim, n_heads, dropout=dropout, batch_first=True,
                kdim=memory_dim, vdim=memory_dim,
            )
            self.norm2 = nn.LayerNorm(dim)
            self.dropout2 = nn.Dropout(dropout)
            if use_gated:
                self.gate_proj_cross = nn.Linear(dim, dim, bias=True)

        # Feedforward (always present)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        # Self-attention
        tgt2 = self.norm1(tgt)
        tgt2, self_attn = self.self_attn(tgt2, tgt2, tgt2, need_weights=return_weights)
        tgt2 = self.dropout1(tgt2)
        if self.use_gated:
            gate = torch.sigmoid(self.gate_proj_self(tgt))
            tgt2 = tgt2 * gate
        tgt = tgt + tgt2

        # Cross-attention to memory (if enabled and memory available)
        cross_attn = None
        if self.use_memory and memory is not None:
            tgt2 = self.norm2(tgt)
            memory_k = memory + memory_pos if memory_pos is not None else memory
            tgt2, cross_attn = self.cross_attn(tgt2, memory_k, memory, need_weights=return_weights)
            tgt2 = self.dropout2(tgt2)
            if self.use_gated:
                gate = torch.sigmoid(self.gate_proj_cross(tgt))
                tgt2 = tgt2 * gate
            tgt = tgt + tgt2

        # Feedforward
        tgt2 = self.norm3(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + self.dropout3(tgt2)

        if return_weights:
            return tgt, self_attn, cross_attn
        return tgt


class MemoryConditionedTransformer(nn.Module):
    """Transformer predictor with optional cross-attention to memory bank."""

    def __init__(
        self,
        dim: int,
        n_blocks: int,
        n_heads: int,
        memory_dim: Optional[int] = None,
        use_memory: bool = True,
        use_gated: bool = False,
        dropout: float = 0.0,
        activation: str = "relu",
        hidden_dim: Optional[int] = None,
        frozen: bool = False,
        **kwargs,  # Absorb extra config arguments for compatibility
    ):
        super().__init__()
        self.use_memory = use_memory

        if hidden_dim is None:
            hidden_dim = 4 * dim
        if memory_dim is None:
            memory_dim = dim

        self.dim = dim
        self.memory_dim = memory_dim

        # Memory-conditioned layers
        self.layers = nn.ModuleList(
            [
                MemoryConditionedLayer(
                    dim=dim,
                    memory_dim=memory_dim,
                    n_heads=n_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                    use_memory=use_memory,
                    use_gated=use_gated,
                )
                for _ in range(n_blocks)
            ]
        )

        self.norm = nn.LayerNorm(dim)

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def forward(
        self,
        slots: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        """
        Predict next frame's slot initialization conditioned on memory.
        
        Args:
            slots: [B, n_slots, dim] - current frame's slots
            memory: [B, N_mem, memory_dim] - concatenated memory features
            memory_pos: [B, N_mem, memory_dim] - temporal positional encodings
            return_weights: bool - whether to return attention weights
            
        Returns:
            predicted_slots: [B, n_slots, dim] - initialization for next frame
            (optional) self_attn_list, cross_attn_list: attention weights from each layer
        """
        output = slots
        self_attn_list = [] if return_weights else None
        cross_attn_list = [] if return_weights else None

        for layer in self.layers:
            if self.use_memory:
                if return_weights:
                    output, self_attn, cross_attn = layer(output, memory, memory_pos, return_weights=True)
                    self_attn_list.append(self_attn)
                    cross_attn_list.append(cross_attn)
                else:
                    output = layer(output, memory, memory_pos)
            else:
                # Ablation: no memory, just self-attention
                if return_weights:
                    output, self_attn, _ = layer(output, memory=None, memory_pos=None, return_weights=True)
                    self_attn_list.append(self_attn)
                else:
                    output = layer(output, memory=None, memory_pos=None)

        if return_weights:
            return self.norm(output), self_attn_list, cross_attn_list
        return self.norm(output)


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    """Like torch.nn.TransformerDecoderLayer, but with customizations."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dim_attn: Optional[int] = None,
        dim_kv: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = torch.nn.functional.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        initial_residual_scale: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device=device,
            dtype=dtype,
        )
        self.self_attn = Attention(
            dim=d_model,
            num_heads=nhead,
            inner_dim=dim_attn,
            qkv_bias=qkv_bias,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.multihead_attn = Attention(
            dim=d_model,
            num_heads=nhead,
            kdim=dim_kv,
            vdim=dim_kv,
            inner_dim=dim_attn,
            qkv_bias=qkv_bias,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        if initial_residual_scale is not None:
            self.scale1 = utils.LayerScale(d_model, init_values=initial_residual_scale)
            self.scale2 = utils.LayerScale(d_model, init_values=initial_residual_scale)
            self.scale3 = utils.LayerScale(d_model, init_values=initial_residual_scale)
        else:
            self.scale1 = nn.Identity()
            self.scale2 = nn.Identity()
            self.scale3 = nn.Identity()

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        x, attn = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_weights=return_weights,
        )
        x = self.dropout1(x)

        if return_weights:
            return x, attn
        else:
            return x, None

    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        return_weights: bool = False,
    ) -> torch.Tensor:
        x, attn = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_weights=return_weights,
        )
        x = self.dropout2(x)

        if return_weights:
            return x, attn
        else:
            return x, None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        x = tgt
        if self.norm_first:
            residual, attn1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self.scale1(residual)
            residual, attn2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask, return_weights
            )
            x = x + self.scale2(residual)
            residual = self._ff_block(self.norm3(x))
            x = x + self.scale3(residual)
        else:
            residual, attn1 = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            x = self.norm1(x + self.scale1(residual))
            residual, attn2 = self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask, return_weights
            )
            x = self.norm2(x + self.scale2(residual))
            residual = self._ff_block(x)
            x = self.norm3(x + self.scale3(residual))

        if return_weights:
            return x, attn1, attn2
        else:
            return x, None, None


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_blocks: int,
        n_heads: int,
        qkv_dim: Optional[int] = None,
        memory_dim: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "relu",
        hidden_dim: Optional[int] = None,
        initial_residual_scale: Optional[float] = None,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.blocks = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    dim,
                    n_heads,
                    dim_feedforward=hidden_dim,
                    dim_attn=qkv_dim,
                    dim_kv=memory_dim,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=1e-05,
                    batch_first=True,
                    norm_first=True,
                    initial_residual_scale=initial_residual_scale,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        output = tgt

        for idx, block in enumerate(self.blocks):
            output, _, attn = block(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                return_weights=return_weights and idx == len(self.blocks) - 1,
            )

        if return_weights:
            return output, attn
        else:
            return output


class HungarianPredictor(nn.Module):
    """Aligns slots temporally using Hungarian matching on slot similarity.

    No learnable parameters - purely matching-based predictor.
    Solves the slot permutation problem by finding optimal 1-to-1 matching
    between consecutive frames based on cosine similarity.

    .. note::
       **Active TPAMI rescue path uses `use_kalman: false` (Round-32 audit).**
       The `use_kalman`, `use_velocity`, and `use_hybrid_cost` flags are
       ablation knobs for the GCv1 oral cycle; all rescue15k configs in
       v2/, phase_a/, phase_b/, and the F.4 / YT-VIS-LR-sweep grids set
       these to False, so the corresponding Kalman / velocity / IoU /
       angular branches are dead code in the published TPAMI results.
       First-frame initialization sets state to None and uses appearance-
       only matching — that is the documented standard, not a Kalman
       "fallback".

    Matching modes (pre_match):
    - False (default): Match AFTER slot attention (original behavior)
    - True: Match BEFORE slot attention, reference is slot attention output
    - "greedy": Match greedy→greedy (same space), reference is matched greedy init
    """

    def __init__(self, dim: int, similarity: str = "cosine", pre_match: bool = False,
                 use_hybrid_cost: bool = False, lambda_pos: float = 1.0,
                 lambda_angular: float = 0.1, lambda_iou: float = 0.0,
                 use_velocity: bool = False, velocity_ema: float = 0.5,
                 use_kalman: bool = False, kalman_process_noise: float = 0.03,
                 use_iterative: bool = False, **kwargs):
        """
        Args:
            dim: Slot dimension (for interface compatibility, not used internally)
            similarity: Similarity metric - 'cosine' or 'l2'
            pre_match: False=post-match, True=pre-match (greedy→slot), "greedy"=greedy→greedy
            use_hybrid_cost: Enable hybrid cost combining appearance, position, angular
            lambda_pos: Weight for position (centroid distance) cost
            lambda_angular: Weight for angular (motion consistency) cost
            lambda_iou: Weight for IoU (mask overlap) cost
            use_velocity: Enable velocity prediction for position matching (simple EMA)
            velocity_ema: EMA weight for velocity smoothing (higher = more smoothing)
            use_kalman: Enable Kalman filter with adaptive uncertainty (overrides use_velocity)
            kalman_process_noise: Process noise for Kalman (motion uncertainty growth rate)
            use_iterative: Use iterative mutual-best matching instead of Hungarian
        """
        super().__init__()
        self.dim = dim
        self.similarity = similarity
        self.pre_match = pre_match
        self.use_hybrid_cost = use_hybrid_cost
        self.lambda_pos = lambda_pos
        self.lambda_angular = lambda_angular
        self.lambda_iou = lambda_iou
        self.use_velocity = use_velocity
        self.velocity_ema = velocity_ema
        self.use_kalman = use_kalman
        self.kalman_process_noise = kalman_process_noise
        self.use_iterative = use_iterative
        self._prev_slots: Optional[torch.Tensor] = None
        self._prev_greedy: Optional[torch.Tensor] = None  # For greedy→greedy matching
        self._last_match_indices: Optional[torch.Tensor] = None  # [B, N] matching indices
        self._last_cost_margin: Optional[torch.Tensor] = None  # [B, N] runner-up minus chosen cost
        # Hybrid cost state
        self._prev_centroids: Optional[torch.Tensor] = None
        self._prev_prev_centroids: Optional[torch.Tensor] = None
        self._velocities: Optional[torch.Tensor] = None  # [B, N, 2] velocity vectors (EMA mode)
        self._prev_masks: Optional[torch.Tensor] = None  # [B, N, H, W] previous masks
        # Kalman state (simplified: position + velocity + scalar uncertainty)
        self._kalman_pos: Optional[torch.Tensor] = None  # [B, N, 2]
        self._kalman_vel: Optional[torch.Tensor] = None  # [B, N, 2]
        self._kalman_uncertainty: Optional[torch.Tensor] = None  # [B, N]

    def reset(self):
        """Reset state for new video sequence."""
        self._prev_slots = None
        self._prev_greedy = None
        self._last_match_indices = None
        self._last_cost_margin = None
        self._prev_centroids = None
        self._prev_prev_centroids = None
        self._velocities = None
        self._prev_masks = None
        self._kalman_pos = None
        self._kalman_vel = None
        self._kalman_uncertainty = None
    
    def get_last_match_indices(self) -> Optional[torch.Tensor]:
        """Return last matching indices [B, N] where indices[b, i] = source slot for position i."""
        return self._last_match_indices

    def match_to_reference(self, slots: torch.Tensor) -> torch.Tensor:
        """Match input slots to stored reference (for pre-matching before slot attention)."""
        if self.pre_match == "greedy":
            # Greedy→greedy: match to previous greedy init, store matched as new reference
            if self._prev_greedy is None:
                self._prev_greedy = slots.detach()
                self._last_match_indices = None
                return slots
            matched, indices = self._hungarian_match(self._prev_greedy, slots, return_indices=True)
            self._prev_greedy = matched.detach()
            self._last_match_indices = indices
            return matched
        else:
            # Original pre_match: match to previous slot attention output
            if self._prev_slots is None:
                self._last_match_indices = None
                return slots
            matched, indices = self._hungarian_match(self._prev_slots, slots, return_indices=True)
            self._last_match_indices = indices
            return matched

    def forward(
        self,
        slots: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        # Hybrid cost inputs (optional, for backward compatibility)
        centroids: Optional[torch.Tensor] = None,
        prev_centroids: Optional[torch.Tensor] = None,
        prev_prev_centroids: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        prev_masks: Optional[torch.Tensor] = None,
        # Absorb unused kwargs (features/flow) passed by LatentProcessor since
        # the Phase 1.0 plumbing patch — HungarianPredictor ignores them.
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            slots: Current frame slots [B, N, D]
            prev_slots: Previous frame slots [B, N, D] (optional, uses internal state if None)
            existence_mask: [B, N] which slots are valid (will be reordered along with slots)
            return_weights: Whether to return matching weights (for interface compatibility)
            centroids: [B, N, 2] current slot centroids (for hybrid cost)
            prev_centroids: [B, N, 2] previous centroids (for hybrid cost)
            prev_prev_centroids: [B, N, 2] centroids from t-2 (for angular cost)
            masks: [B, N, H, W] current slot masks (for IoU cost)
            prev_masks: [B, N, H, W] previous masks (for IoU cost)
        
        Returns:
            Reordered slots to match previous frame's slot ordering [B, N, D]
            If existence_mask provided: (reordered_slots, reordered_mask, None)
        """
        if self.pre_match == "greedy":
            # Greedy mode: matching done in match_to_reference, just pass through
            if existence_mask is not None:
                return slots, existence_mask, None
            if return_weights:
                return slots, None
            return slots
        
        if self.pre_match:
            # Pre-match mode: matching was done before slot attention, update slot reference
            self._prev_slots = slots.detach()
            if existence_mask is not None:
                return slots, existence_mask, None
            if return_weights:
                return slots, None
            return slots
        
        # Post-match mode (original behavior)
        reference_slots = prev_slots if prev_slots is not None else self._prev_slots
        
        if reference_slots is None:
            self._prev_slots = slots.detach()
            self._last_match_indices = None
            # Initialize hybrid cost state
            if self.use_hybrid_cost:
                self._prev_centroids = centroids.detach() if centroids is not None else None
                self._prev_prev_centroids = None
            if existence_mask is not None:
                return slots, existence_mask, None
            if return_weights:
                return slots, None
            return slots
        
        # Use cached prev values if not provided
        if self.use_hybrid_cost:
            if prev_centroids is None:
                prev_centroids = self._prev_centroids
            if prev_prev_centroids is None:
                prev_prev_centroids = self._prev_prev_centroids
            if prev_masks is None:
                prev_masks = self._prev_masks
        
        reordered_slots, indices = self._hungarian_match(
            reference_slots, slots, return_indices=True,
            prev_centroids=prev_centroids if self.use_hybrid_cost else None,
            curr_centroids=centroids if self.use_hybrid_cost else None,
            prev_prev_centroids=prev_prev_centroids if self.use_hybrid_cost else None,
            prev_masks=prev_masks if self.use_hybrid_cost else None,
            curr_masks=masks if self.use_hybrid_cost else None,
        )
        self._prev_slots = reordered_slots.detach()
        self._last_match_indices = indices
        
        # Update Kalman filter or simple velocity
        if self.use_kalman and centroids is not None and prev_centroids is not None:
            B = centroids.shape[0]
            matched_centroids = torch.stack([centroids[b, indices[b]] for b in range(B)], dim=0)
            
            if self._kalman_pos is None:
                # Initialize Kalman state
                self._kalman_pos = prev_centroids.detach()
                self._kalman_vel = (matched_centroids - prev_centroids).detach()
                self._kalman_uncertainty = torch.ones(B, centroids.shape[1], device=centroids.device) * 0.1
            else:
                # Predict step
                predicted_pos = self._kalman_pos + self._kalman_vel
                self._kalman_uncertainty = self._kalman_uncertainty + self.kalman_process_noise
                
                # Update step
                innovation = matched_centroids - predicted_pos
                kalman_gain = self._kalman_uncertainty / (self._kalman_uncertainty + 0.1)
                self._kalman_pos = predicted_pos + kalman_gain.unsqueeze(-1) * innovation
                self._kalman_vel = self._kalman_vel + 0.5 * kalman_gain.unsqueeze(-1) * innovation
                self._kalman_uncertainty = (1 - kalman_gain) * self._kalman_uncertainty
                
        elif self.use_velocity and centroids is not None and prev_centroids is not None:
            B = centroids.shape[0]
            matched_centroids = torch.stack([centroids[b, indices[b]] for b in range(B)], dim=0)
            new_v = matched_centroids - prev_centroids
            if self._velocities is not None:
                self._velocities = self.velocity_ema * self._velocities + (1 - self.velocity_ema) * new_v
            else:
                self._velocities = new_v
        
        # Update hybrid cost state history
        if self.use_hybrid_cost:
            self._prev_prev_centroids = self._prev_centroids
            self._prev_centroids = centroids.detach() if centroids is not None else None
            self._prev_masks = masks.detach() if masks is not None else None
        
        # Reorder existence_mask if provided
        if existence_mask is not None:
            B = slots.shape[0]
            reordered_mask = torch.stack([existence_mask[b, indices[b]] for b in range(B)], dim=0)
            return reordered_slots, reordered_mask, None
        
        if return_weights:
            return reordered_slots, None
        return reordered_slots

    def _compute_hybrid_cost(
        self,
        prev_slots: torch.Tensor,
        curr_slots: torch.Tensor,
        prev_centroids: Optional[torch.Tensor] = None,
        curr_centroids: Optional[torch.Tensor] = None,
        prev_prev_centroids: Optional[torch.Tensor] = None,
        prev_masks: Optional[torch.Tensor] = None,
        curr_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute hybrid cost matrix combining appearance, position, angular, and IoU costs.
        
        Args:
            prev_slots: [B, N, D]
            curr_slots: [B, N, D]
            prev_centroids: [B, N, 2] previous centroids
            curr_centroids: [B, N, 2] current centroids
            prev_prev_centroids: [B, N, 2] centroids from t-2 for angular cost
            prev_masks: [B, N, H, W] previous masks
            curr_masks: [B, N, H, W] current masks
        
        Returns:
            cost_matrix: [B, N, N] hybrid cost matrix
        """
        B, N, D = curr_slots.shape
        
        # 1. Appearance cost (base cost - always computed, normalized to [0, 1])
        if self.similarity == "cosine":
            prev_norm = F.normalize(prev_slots, dim=-1)
            curr_norm = F.normalize(curr_slots, dim=-1)
            sim_matrix = torch.bmm(prev_norm, curr_norm.transpose(1, 2))
            # Normalize: cosine_sim ∈ [-1, 1] → cost ∈ [0, 1]
            appearance_cost = (1 - sim_matrix) / 2  # [B, N, N]
        else:  # L2 distance
            diff = prev_slots.unsqueeze(2) - curr_slots.unsqueeze(1)
            distance = diff.norm(dim=-1)
            # Normalize using RBF kernel: scale by sqrt(2*D) for typical feature scale
            # This maps [0, ∞) → [0, 1] smoothly
            scale = (2 * D) ** 0.5
            appearance_cost = 1 - torch.exp(-distance / scale)
        
        cost_matrix = appearance_cost
        
        # 2. Position cost (centroid distance with velocity/Kalman prediction)
        if self.lambda_pos > 0 and prev_centroids is not None and curr_centroids is not None:
            ref_centroids = prev_centroids
            uncertainty_weight = 1.0
            
            if self.use_kalman and self._kalman_pos is not None:
                # Kalman prediction: position + velocity
                ref_centroids = self._kalman_pos + self._kalman_vel
                # Adaptive weighting: high uncertainty → less trust in position
                uncertainty_weight = 1.0 + self._kalman_uncertainty.unsqueeze(2)  # [B, N, 1]
            elif self.use_velocity and self._velocities is not None:
                ref_centroids = prev_centroids + self._velocities
            
            # [B, N, 1, 2] - [B, 1, N, 2] -> [B, N, N]
            spatial_diff = ref_centroids.unsqueeze(2) - curr_centroids.unsqueeze(1)
            spatial_cost = spatial_diff.norm(dim=-1)  # [B, N, N]
            # Normalize by sqrt(2) (max distance for normalized [0,1] coords)
            spatial_cost = spatial_cost / (2 ** 0.5)
            # Apply uncertainty weighting (Kalman mode only)
            spatial_cost = spatial_cost * uncertainty_weight
            cost_matrix = cost_matrix + self.lambda_pos * spatial_cost
        
        # 3. Angular cost (motion consistency)
        if (self.lambda_angular > 0 and prev_centroids is not None and 
            curr_centroids is not None and prev_prev_centroids is not None):
            # Motion vector from t-2 to t-1
            prev_motion = prev_centroids - prev_prev_centroids  # [B, N, 2]
            # Candidate motion vectors from t-1 to t (for each matching)
            # [B, N, 1, 2] - [B, 1, N, 2] -> [B, N, N, 2]
            curr_motion = curr_centroids.unsqueeze(1) - prev_centroids.unsqueeze(2)
            
            # Cosine distance between motion vectors
            prev_motion_norm = F.normalize(prev_motion, dim=-1, eps=1e-8)  # [B, N, 2]
            curr_motion_norm = F.normalize(curr_motion, dim=-1, eps=1e-8)  # [B, N, N, 2]
            # Dot product: [B, N, 1, 2] * [B, N, N, 2] -> [B, N, N]
            cosine_sim = (prev_motion_norm.unsqueeze(2) * curr_motion_norm).sum(dim=-1)
            # Normalize: cosine_sim ∈ [-1, 1] → cost ∈ [0, 1]
            angular_cost = (1 - cosine_sim) / 2  # [B, N, N]
            cost_matrix = cost_matrix + self.lambda_angular * angular_cost
        
        # 4. IoU cost (mask overlap - feature-independent spatial matching)
        if self.lambda_iou > 0 and prev_masks is not None and curr_masks is not None:
            # Reshape 3D masks [B, N, n_patches] to 4D [B, N, H, W] if needed
            # (SlotAttention returns 3D masks, but IoU needs pairwise spatial overlap)
            if prev_masks.ndim == 3:
                h = int(prev_masks.shape[-1] ** 0.5)
                prev_masks = prev_masks.view(prev_masks.shape[0], prev_masks.shape[1], h, h)
            if curr_masks.ndim == 3:
                h = int(curr_masks.shape[-1] ** 0.5)
                curr_masks = curr_masks.view(curr_masks.shape[0], curr_masks.shape[1], h, h)
            # Check dimensions match (masks might have different N than slots)
            if prev_masks.shape[1] == N and curr_masks.shape[1] == N:
                # prev_masks: [B, N, H, W], curr_masks: [B, N, H, W]
                prev_m = prev_masks.unsqueeze(2)  # [B, N, 1, H, W]
                curr_m = curr_masks.unsqueeze(1)  # [B, 1, N, H, W]
                intersection = (prev_m * curr_m).sum(dim=(-2, -1))  # [B, N, N]
                union = (prev_m + curr_m - prev_m * curr_m).sum(dim=(-2, -1))  # [B, N, N]
                iou = intersection / (union + 1e-8)  # [B, N, N]
                cost_matrix = cost_matrix + self.lambda_iou * (1 - iou)
        
        return cost_matrix

    def _iterative_match(
        self, prev_slots: torch.Tensor, curr_slots: torch.Tensor, return_indices: bool = False,
        prev_centroids: Optional[torch.Tensor] = None,
        curr_centroids: Optional[torch.Tensor] = None,
        prev_prev_centroids: Optional[torch.Tensor] = None,
        prev_masks: Optional[torch.Tensor] = None,
        curr_masks: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Iterative mutual-best matching with Hungarian fallback.
        
        Phase 1: Lock in confident matches where i is j's minimum AND j is i's minimum
        Phase 2: Hungarian on remaining unmatched slots to ensure complete bijection
        
        Args:
            prev_slots: Reference slots from previous frame [B, N, D]
            curr_slots: Current slots to reorder [B, N, D]
            return_indices: If True, also return matching indices
            prev_centroids, curr_centroids: Optional centroids for hybrid cost
            prev_prev_centroids: Optional t-2 centroids for angular cost
        
        Returns:
            Reordered curr_slots [B, N, D], and optionally indices [B, N]
        """
        from scipy.optimize import linear_sum_assignment
        
        B, N, D = curr_slots.shape
        device = curr_slots.device
        
        # Compute cost matrix (reuse existing logic)
        if self.use_hybrid_cost:
            cost_matrix = self._compute_hybrid_cost(
                prev_slots, curr_slots, prev_centroids, curr_centroids, prev_prev_centroids,
                prev_masks, curr_masks
            )
        else:
            if self.similarity == "cosine":
                prev_norm = F.normalize(prev_slots, dim=-1)
                curr_norm = F.normalize(curr_slots, dim=-1)
                sim_matrix = torch.bmm(prev_norm, curr_norm.transpose(1, 2))
                cost_matrix = 1 - sim_matrix
            else:
                diff = prev_slots.unsqueeze(2) - curr_slots.unsqueeze(1)
                cost_matrix = diff.norm(dim=-1)
        
        # Process each batch element
        reordered_list = []
        indices_list = []
        
        for b in range(B):
            cost = cost_matrix[b]  # [N, N]
            
            # Phase 1: Iterative mutual-best matching
            unmatched_prev = set(range(N))
            unmatched_curr = set(range(N))
            matches = {}  # prev_idx -> curr_idx
            
            while True:
                if len(unmatched_prev) == 0:
                    break
                
                # For each unmatched prev, find best curr
                prev_to_curr = {}  # prev_idx -> best_curr_idx
                for i in unmatched_prev:
                    best_j = None
                    best_cost = float('inf')
                    for j in unmatched_curr:
                        if cost[i, j] < best_cost:
                            best_cost = cost[i, j]
                            best_j = j
                    if best_j is not None:
                        prev_to_curr[i] = best_j
                
                # For each unmatched curr, find best prev
                curr_to_prev = {}  # curr_idx -> best_prev_idx
                for j in unmatched_curr:
                    best_i = None
                    best_cost = float('inf')
                    for i in unmatched_prev:
                        if cost[i, j] < best_cost:
                            best_cost = cost[i, j]
                            best_i = i
                    if best_i is not None:
                        curr_to_prev[j] = best_i
                
                # Find mutual bests: i->j and j->i
                mutual_best = []
                for i in unmatched_prev:
                    if i in prev_to_curr:
                        j = prev_to_curr[i]
                        if j in curr_to_prev and curr_to_prev[j] == i:
                            mutual_best.append((i, j))
                
                if len(mutual_best) == 0:
                    break  # No more mutual-best pairs
                
                # Lock in mutual-best matches
                for i, j in mutual_best:
                    matches[i] = j
                    unmatched_prev.remove(i)
                    unmatched_curr.remove(j)
            
            # Phase 2: Hungarian on remaining unmatched
            if len(unmatched_prev) > 0:
                # Extract submatrix
                unmatched_prev_list = sorted(list(unmatched_prev))
                unmatched_curr_list = sorted(list(unmatched_curr))
                
                sub_cost = cost[unmatched_prev_list][:, unmatched_curr_list]
                sub_cost_np = sub_cost.detach().cpu().numpy()
                
                row_ind, col_ind = linear_sum_assignment(sub_cost_np)
                
                # Map back to original indices
                for r, c in zip(row_ind, col_ind):
                    i = unmatched_prev_list[r]
                    j = unmatched_curr_list[c]
                    matches[i] = j
            
            # Build reordered output: position i gets curr_slots[matches[i]]
            col_ind = [matches[i] for i in range(N)]
            reordered = curr_slots[b, col_ind]
            reordered_list.append(reordered)
            indices_list.append(torch.tensor(col_ind, device=device, dtype=torch.long))
        
        reordered = torch.stack(reordered_list, dim=0)
        if return_indices:
            indices = torch.stack(indices_list, dim=0)
            return reordered, indices
        return reordered

    def _hungarian_match(
        self, prev_slots: torch.Tensor, curr_slots: torch.Tensor, return_indices: bool = False,
        prev_centroids: Optional[torch.Tensor] = None,
        curr_centroids: Optional[torch.Tensor] = None,
        prev_prev_centroids: Optional[torch.Tensor] = None,
        prev_masks: Optional[torch.Tensor] = None,
        curr_masks: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply matching to align curr_slots with prev_slots ordering.
        
        Uses iterative mutual-best matching if use_iterative=True, else standard Hungarian.
        
        Args:
            prev_slots: Reference slots from previous frame [B, N, D]
            curr_slots: Current slots to reorder [B, N, D]
            return_indices: If True, also return matching indices
            prev_centroids, curr_centroids: Optional centroids for hybrid cost
            prev_prev_centroids: Optional t-2 centroids for angular cost
        
        Returns:
            Reordered curr_slots [B, N, D], and optionally indices [B, N]
        """
        # Dispatch to iterative matching if enabled
        if self.use_iterative:
            return self._iterative_match(
                prev_slots, curr_slots, return_indices,
                prev_centroids, curr_centroids, prev_prev_centroids,
                prev_masks, curr_masks
            )
        
        # Standard Hungarian matching
        from scipy.optimize import linear_sum_assignment
        
        B, N, D = curr_slots.shape
        device = curr_slots.device
        
        # Compute cost matrix
        if self.use_hybrid_cost:
            cost_matrix = self._compute_hybrid_cost(
                prev_slots, curr_slots, prev_centroids, curr_centroids,
                prev_prev_centroids, prev_masks, curr_masks
            )
        else:
            # Original cost computation (backward compatible)
            if self.similarity == "cosine":
                prev_norm = F.normalize(prev_slots, dim=-1)
                curr_norm = F.normalize(curr_slots, dim=-1)
                sim_matrix = torch.bmm(prev_norm, curr_norm.transpose(1, 2))
                cost_matrix = 1 - sim_matrix
            else:  # L2 distance
                diff = prev_slots.unsqueeze(2) - curr_slots.unsqueeze(1)
                cost_matrix = diff.norm(dim=-1)
        
        # Apply Hungarian algorithm per batch element
        reordered_list = []
        indices_list = []
        margins = []
        for b in range(B):
            cost_np = cost_matrix[b].detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            # col_ind tells us which curr_slot should go to which position
            reordered = curr_slots[b, col_ind]  # [N, D]
            reordered_list.append(reordered)
            indices_list.append(torch.tensor(col_ind, device=device, dtype=torch.long))
            # Cost-margin diagnostic (Round 20 fix #4): for each row, compare the
            # chosen-column cost vs the runner-up cost (min over remaining cols).
            # margin > 0 ⇒ Hungarian had a non-trivial preference; margin ≈ 0
            # could mean tie-break / arbitrary. Stored as [N] tensor per batch.
            chosen = cost_np[row_ind, col_ind]  # [N]
            cost_masked = cost_np.copy()
            cost_masked[row_ind, col_ind] = float("inf")
            runner_up = cost_masked.min(axis=1)  # [N]
            margins.append(torch.tensor(runner_up - chosen, device=device, dtype=torch.float32))

        reordered = torch.stack(reordered_list, dim=0)  # [B, N, D]
        self._last_cost_margin = torch.stack(margins, dim=0)  # [B, N]
        if return_indices:
            indices = torch.stack(indices_list, dim=0)  # [B, N]
            return reordered, indices
        return reordered


class SoftIdentityPredictor(nn.Module):
    """Differentiable identity-aware predictor (Round-24 Hail Mary).

    Codex Round-3 prescription for Claim 3 (LoRA replaces hard Hungarian).
    Unlike :class:`HungarianPredictor` whose argmax matching has zero
    gradient, this module emits a soft assignment matrix ``A [B, N, N]``
    via Sinkhorn normalization of an affinity matrix that combines

    1. Cosine similarity between previous and current slot vectors, and
    2. Flow-warped soft IoU between previous and current slot masks
       (when ``forward_flow`` and ``slot_masks`` are available).

    The output ``A @ current_slots`` reorders current slots into the
    previous-frame identity order (matching the convention in
    :class:`HungarianPredictor.forward`).  Because Sinkhorn is fully
    differentiable, gradients from any downstream loss flow back through
    the assignment into both the current slot vectors and (importantly)
    the LoRA-parameterised backbone -- so the network can learn to make
    its own slots matchable across frames, replacing hard Hungarian.

    The module exposes the same sentinel attribute (``_hungarian_match``)
    that :class:`LatentProcessor` uses to detect Hungarian-family
    predictors, so it slots in via the existing dispatch in
    ``slotcontrast/modules/video.py`` without further plumbing changes.
    """

    def __init__(
        self,
        dim: int,
        sinkhorn_iters: int = 5,
        sinkhorn_tau: float = 0.1,
        cosine_weight: float = 1.0,
        flow_iou_weight: float = 1.0,
        use_flow_iou: bool = True,
        # Centroid-distance affinity (added Round-24 follow-on after the cosine-
        # only sweep showed Claim 3 partial). Provides a spatial-prior signal
        # without requiring forward_flow plumbing in the dataset pipeline.
        use_centroid_dist: bool = False,
        centroid_dist_weight: float = 1.0,
        centroid_dist_temp: float = 0.1,
        # Straight-through Hungarian (Round-26 architectural fusion).
        # When True, forward output is a HARD permutation of slots (argmax of
        # the Sinkhorn assignment, gathered into an identity-permuted tensor)
        # while gradients flow through the soft Sinkhorn matrix (Mena et al.,
        # 2018, "Learning Latent Permutations with Gumbel-Sinkhorn Networks").
        # Combines the discrete semantics that win for hard-Hungarian on
        # MOVi-D with the gradient flow that wins for soft Sinkhorn on
        # MOVi-E. Default False preserves the soft-mixture output.
        straight_through: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_tau = float(sinkhorn_tau)
        self.cosine_weight = float(cosine_weight)
        self.flow_iou_weight = float(flow_iou_weight)
        self.use_flow_iou = bool(use_flow_iou)
        self.use_centroid_dist = bool(use_centroid_dist)
        self.centroid_dist_weight = float(centroid_dist_weight)
        self.centroid_dist_temp = float(centroid_dist_temp)
        self.straight_through = bool(straight_through)
        # State (cleared via reset() between videos)
        self._prev_slots: Optional[torch.Tensor] = None
        self._prev_masks: Optional[torch.Tensor] = None
        self._prev_flow: Optional[torch.Tensor] = None
        self._prev_centroids: Optional[torch.Tensor] = None
        self._last_assignment: Optional[torch.Tensor] = None
        self._last_match_indices: Optional[torch.Tensor] = None
        self._last_cost_margin: Optional[torch.Tensor] = None
        # Sentinel flags expected by LatentProcessor's dispatch
        self._hungarian_match = None  # marker only; not called
        self.pre_match = False        # post-match style (output reordered)

    def reset(self) -> None:
        self._prev_slots = None
        self._prev_masks = None
        self._prev_flow = None
        self._prev_centroids = None
        self._last_assignment = None
        self._last_match_indices = None
        self._last_cost_margin = None

    def get_last_match_indices(self) -> Optional[torch.Tensor]:
        return self._last_match_indices

    @staticmethod
    def _sinkhorn(log_alpha: torch.Tensor, n_iters: int) -> torch.Tensor:
        # Doubly-stochastic normalization in log-space. log_alpha: [B, N, N].
        x = log_alpha
        for _ in range(n_iters):
            x = x - torch.logsumexp(x, dim=-1, keepdim=True)
            x = x - torch.logsumexp(x, dim=-2, keepdim=True)
        return x.exp()

    @staticmethod
    def _mask_centroids(masks_4d: torch.Tensor) -> torch.Tensor:
        # Compute spatial centroids from [B,N,H,W] masks; returns [B,N,2] in
        # normalized (h, w) coords on [0, 1]. Mirrors the convention in
        # ``initializers.compute_slot_centroids_from_masks`` so soft-IoU and
        # centroid-distance affinities live on a comparable scale.
        B, N, H, W = masks_4d.shape
        device = masks_4d.device
        coords_h = torch.arange(H, device=device, dtype=torch.float32)
        coords_w = torch.arange(W, device=device, dtype=torch.float32)
        grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing="ij")
        m = masks_4d.float()
        weight = m.flatten(-2).sum(-1).clamp_min(1e-8)               # [B, N]
        ch = (m * grid_h.view(1, 1, H, W)).flatten(-2).sum(-1) / weight
        cw = (m * grid_w.view(1, 1, H, W)).flatten(-2).sum(-1) / weight
        ch = ch / max(H - 1, 1)
        cw = cw / max(W - 1, 1)
        return torch.stack([ch, cw], dim=-1)

    @staticmethod
    def _reshape_masks_to_4d(masks: torch.Tensor) -> torch.Tensor:
        # Convert [B,N,P] (patch-flattened, square) -> [B,N,H,W]; pass through
        # [B,N,H,W]. Hard-raise on any other shape (no silent fallback).
        if masks.dim() == 4:
            return masks
        if masks.dim() == 3:
            P = masks.shape[-1]
            side = int(round(P ** 0.5))
            if side * side != P:
                raise ValueError(
                    f"SoftIdentityPredictor: cannot reshape patch-flattened mask "
                    f"with P={P} (sqrt({P}) is not integer); spatial dim required."
                )
            B, N = masks.shape[:2]
            return masks.reshape(B, N, side, side)
        raise ValueError(
            f"SoftIdentityPredictor: slot_masks must be [B,N,P] or [B,N,H,W], "
            f"got shape {tuple(masks.shape)}."
        )

    @staticmethod
    def _flow_warp_masks(
        prev_masks: torch.Tensor, flow: torch.Tensor
    ) -> torch.Tensor:
        # prev_masks: [B, N, H, W]; flow: [B, Hf, Wf, 2] (dx,dy in pixels at
        # native resolution) OR [B, 2, Hf, Wf]. Returns prev_masks resampled
        # into the current-frame coordinate system using a backward-sample
        # warp (sample prev at curr - flow). Hard-raise on shape errors.
        B, N, H, W = prev_masks.shape
        if flow.dim() != 4:
            raise ValueError(
                f"SoftIdentityPredictor: forward_flow must be 4D "
                f"[B,Hf,Wf,2] or [B,2,Hf,Wf]; got shape {tuple(flow.shape)}."
            )
        if flow.shape[-1] == 2:
            flow_bcyx = flow.permute(0, 3, 1, 2).contiguous()  # [B, 2, Hf, Wf]
            Hf, Wf = flow.shape[1], flow.shape[2]
        elif flow.shape[1] == 2:
            flow_bcyx = flow
            Hf, Wf = flow.shape[2], flow.shape[3]
        else:
            raise ValueError(
                f"SoftIdentityPredictor: forward_flow channel layout unrecognized; "
                f"got shape {tuple(flow.shape)}."
            )
        if (Hf, Wf) != (H, W):
            flow_bcyx = F.interpolate(
                flow_bcyx.float(), size=(H, W), mode="bilinear",
                align_corners=False,
            )
            # Rescale dx,dy from native (Hf,Wf) pixel units to mask (H,W) units
            scale_x = W / max(Wf, 1)
            scale_y = H / max(Hf, 1)
            flow_bcyx = flow_bcyx * torch.tensor(
                [scale_x, scale_y], device=prev_masks.device, dtype=flow_bcyx.dtype,
            ).view(1, 2, 1, 1)
        # Build sampling grid using the same convention as
        # `FlowConsistencyMaskLoss._build_sampling_grid` — align_corners=False
        # with half-pixel destination centres. We move flow to [B,H,W,2]
        # ``(dx, dy)`` and reuse the analytical formula:
        #   src = dest - flow_norm,   flow_norm = flow_pixels * (2/W, 2/H).
        device = prev_masks.device
        dtype_f = torch.float32
        ys = (torch.arange(H, device=device, dtype=dtype_f) + 0.5) * (2.0 / H) - 1.0
        xs = (torch.arange(W, device=device, dtype=dtype_f) + 0.5) * (2.0 / W) - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # [1,H,W,2]
        flow_hw2 = flow_bcyx.permute(0, 2, 3, 1).contiguous()           # [B,H,W,2]
        scale = torch.tensor([2.0 / W, 2.0 / H], device=device, dtype=dtype_f)
        grid = base_grid - flow_hw2.float() * scale                     # [B,H,W,2]
        masks_flat = prev_masks.reshape(B * N, 1, H, W).float()
        grid_flat = grid.unsqueeze(1).expand(B, N, H, W, 2).reshape(B * N, H, W, 2)
        warped = F.grid_sample(
            masks_flat, grid_flat, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )
        return warped.reshape(B, N, H, W).to(prev_masks.dtype)

    def match_to_reference(self, slots: torch.Tensor) -> torch.Tensor:
        # Pre-match path is disabled for SoftIdentityPredictor; the slot
        # ordering happens in forward().  This stub keeps interface parity
        # with HungarianPredictor in case a config sets pre_match=True.
        return slots

    def forward(
        self,
        slots: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        slot_masks: Optional[torch.Tensor] = None,
        prev_masks: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B, N, D = slots.shape
        # First call (no internal state): cache and pass through.  We mirror
        # HungarianPredictor's first-frame convention by leaving
        # ``_last_match_indices`` and ``_last_cost_margin`` at None so the
        # ScanOverTime aggregator records ``None`` for frame 0 (otherwise the
        # identity-ratio mean is inflated and the cost-margin mean picks up a
        # spurious zero entry).
        if self._prev_slots is None and prev_slots is None:
            self._prev_slots = slots.detach()
            if slot_masks is not None:
                self._prev_masks = slot_masks.detach()
                if self.use_centroid_dist:
                    self._prev_centroids = self._mask_centroids(
                        self._reshape_masks_to_4d(slot_masks).detach()
                    )
            if flow is not None:
                # Cache flow for use at NEXT step. forward_flow[t] tells motion
                # from t to t+1, which is what we need to warp _prev_masks
                # (captured at step t) into the next frame's coordinates.
                self._prev_flow = flow.detach()
            self._last_assignment = torch.eye(
                N, device=slots.device, dtype=slots.dtype
            ).unsqueeze(0).expand(B, N, N).contiguous()
            self._last_match_indices = None
            self._last_cost_margin = None
            if existence_mask is not None:
                return slots, existence_mask, None
            if return_weights:
                return slots, None
            return slots

        prev = prev_slots.detach() if prev_slots is not None else self._prev_slots
        if prev.shape != slots.shape:
            # Codex Round-27 audit: hard-raise on slot-count mismatch instead of
            # silent identity fallback. Honors the project-wide "no fallback"
            # rule and turns existence-mask-churn bugs loud rather than masking
            # them as soft "matched output equals input" no-ops.
            raise ValueError(
                f"SoftIdentityPredictor: previous slot tensor shape "
                f"{tuple(prev.shape)} does not match current slot tensor shape "
                f"{tuple(slots.shape)}. This usually indicates an existence-mask "
                f"or num-slots mismatch; investigate the upstream initializer."
            )

        # Cosine affinity: [B, N(prev), N(curr)]
        prev_norm = F.normalize(prev.float(), dim=-1)
        curr_norm = F.normalize(slots.float(), dim=-1)
        affinity = self.cosine_weight * torch.einsum(
            "bid,bjd->bij", prev_norm, curr_norm
        )

        # Centroid-distance affinity (optional). Computes per-slot spatial
        # centroids from ``slot_masks`` and ``_prev_masks`` and converts the
        # squared centroid distance into an affinity via
        # ``exp(-d^2 / centroid_dist_temp)``. Provides a position-prior that
        # complements cosine without requiring forward_flow plumbing.
        curr_centroids: Optional[torch.Tensor] = None
        if self.use_centroid_dist:
            if slot_masks is None:
                raise ValueError(
                    "SoftIdentityPredictor: use_centroid_dist=True requires "
                    "slot_masks (got None)."
                )
            cur_masks_4d_for_c = self._reshape_masks_to_4d(slot_masks)
            B_c, N_c, H_c, W_c = cur_masks_4d_for_c.shape
            curr_centroids = self._mask_centroids(cur_masks_4d_for_c)
            if self._prev_centroids is not None:
                # Pairwise squared Euclidean distance, normalized by image size
                # so the temperature is invariant to mask resolution.
                pc = self._prev_centroids                # [B, Nprev, 2]
                cc = curr_centroids                      # [B, Ncurr, 2]
                diff = pc.unsqueeze(2) - cc.unsqueeze(1)  # [B, Nprev, Ncurr, 2]
                d2 = (diff * diff).sum(-1)                # [B, Nprev, Ncurr]
                cent_aff = torch.exp(-d2 / max(self.centroid_dist_temp, 1e-4))
                affinity = affinity + self.centroid_dist_weight * cent_aff.to(affinity.dtype)

        # Flow-warped soft IoU (optional). slot_masks may be 3D [B,N,P] (patch-
        # flattened) or 4D [B,N,H,W]; reshape to 4D for the warp + IoU. We use
        # the *previous* step's flow (cached as ``_prev_flow`` at the end of
        # the prior forward) because forward_flow[t] is the t->t+1 motion and
        # at step t we need the t-1 -> t motion to warp ``_prev_masks`` into
        # the current frame's coordinates. No silent fallbacks: if flow-IoU is
        # configured but the inputs are not consistent we hard-raise so that
        # bad data is loud rather than masked.
        if self.use_flow_iou:
            if slot_masks is None:
                # SoftIdentity is configured for flow-IoU but the dispatcher
                # did not pass slot_masks. This is a config error — the model
                # must produce per-slot masks.
                raise ValueError(
                    "SoftIdentityPredictor: use_flow_iou=True requires "
                    "slot_masks (got None)."
                )
            if self._prev_masks is None:
                # First post-reset step has no cached prev mask yet. Skip the
                # IoU term silently here — the cosine affinity above is still
                # well-defined.
                pass
            elif self._prev_flow is None:
                # use_flow_iou is on, the model has cached prev masks, but no
                # flow was provided this step. Either the dataset omitted
                # forward_flow or the previous forward did not see flow. Hard
                # raise so the missing plumbing is caught loudly.
                raise ValueError(
                    "SoftIdentityPredictor: use_flow_iou=True but no "
                    "forward_flow was supplied at the previous step. Add "
                    "`forward_flow` to the dataset pipeline keys, or set "
                    "`use_flow_iou: false` in the predictor config."
                )
            else:
                if self._prev_masks.shape != slot_masks.shape:
                    raise ValueError(
                        f"SoftIdentityPredictor: prev/curr slot_masks shape "
                        f"mismatch {tuple(self._prev_masks.shape)} vs "
                        f"{tuple(slot_masks.shape)}."
                    )
                cur_masks_4d = self._reshape_masks_to_4d(slot_masks)
                prev_masks_4d = self._reshape_masks_to_4d(self._prev_masks)
                warped = self._flow_warp_masks(
                    prev_masks_4d.float(), self._prev_flow.float()
                )
                inter = torch.einsum(
                    "bihw,bjhw->bij", warped, cur_masks_4d.float()
                )
                area_prev = warped.flatten(2).sum(-1).unsqueeze(-1)
                area_curr = cur_masks_4d.float().flatten(2).sum(-1).unsqueeze(1)
                union = area_prev + area_curr - inter + 1e-6
                iou = inter / union
                affinity = affinity + self.flow_iou_weight * iou.to(affinity.dtype)

        # Sinkhorn -> doubly-stochastic soft assignment.
        log_alpha = affinity / max(self.sinkhorn_tau, 1e-4)
        A = self._sinkhorn(log_alpha, self.sinkhorn_iters)  # [B, Nprev, Ncurr]
        A = A.to(slots.dtype)

        # Reorder current slots into previous identity order.
        # Soft path:  out[b,i] = sum_j A[b,i,j] * slots[b,j]
        # Straight-through path: forward uses a TRUE Hungarian permutation
        # (linear_sum_assignment on -A so high-affinity entries are picked,
        # guaranteeing a bijection — row-wise argmax does NOT give a
        # permutation when two prev slots prefer the same curr slot, codex
        # Round-8 audit). Backward flows through A via the standard
        # ``hard - A.detach() + A`` identity. This combines hard-Hungarian's
        # discrete identity semantics with soft-Sinkhorn's gradient flow.
        if self.straight_through:
            from scipy.optimize import linear_sum_assignment
            A_np = A.detach().cpu().numpy()
            hard_A = torch.zeros_like(A)
            hard_idx_list = []
            for b in range(B):
                row_ind, col_ind = linear_sum_assignment(-A_np[b])
                hard_A[b, row_ind, col_ind] = 1.0
                # row_ind is always 0..N-1 in standard order, so col_ind[i]
                # tells us "current slot index for previous identity i".
                hard_idx_list.append(torch.as_tensor(col_ind, device=A.device, dtype=torch.long))
            hard_idx = torch.stack(hard_idx_list, dim=0)  # [B, N]
            A_used = (hard_A - A).detach() + A             # forward = hard permutation, grad ≈ A
            output = torch.einsum("bij,bjd->bid", A_used, slots)
        else:
            output = torch.einsum("bij,bjd->bid", A, slots)
            hard_idx = None  # only set in straight-through path

        # Diagnostics & state cache. In straight-through mode, the recorded
        # match indices come from the Hungarian/LSAP projection (true
        # bijection), not row-argmax — keeps the identity-ratio diagnostic
        # consistent with the actual forward semantics.
        with torch.no_grad():
            sorted_aff, _ = affinity.sort(dim=-1, descending=True)
            margin = (
                sorted_aff[..., 0] - sorted_aff[..., 1]
                if sorted_aff.shape[-1] >= 2
                else torch.zeros(B, N, device=slots.device)
            )
            self._last_cost_margin = margin
            if self.straight_through:
                self._last_match_indices = hard_idx  # from LSAP
            else:
                self._last_match_indices = A.argmax(dim=-1)
            self._last_assignment = A.detach()

        # Cache reordered prev for next step (detached so gradient does not
        # flow across frames). In straight-through mode use the hard
        # permutation matrix (consistent with the forward semantics) so the
        # cached prev_masks/centroids stay aligned with the discrete identity
        # output the decoder sees, not the soft mixture.
        self._prev_slots = output.detach()
        cache_A = hard_A.detach() if self.straight_through else A.detach()
        if slot_masks is not None:
            # slot_masks may be 3D [B,N,P] or 4D [B,N,H,W]; reorder along N
            # using ``cache_A``. Hard-raise on unsupported ranks (no silent fallback).
            sm = slot_masks.detach()
            if sm.dim() == 3:
                self._prev_masks = torch.einsum(
                    "bij,bjp->bip", cache_A, sm
                )
            elif sm.dim() == 4:
                self._prev_masks = torch.einsum(
                    "bij,bjhw->bihw", cache_A, sm
                )
            else:
                raise ValueError(
                    f"SoftIdentityPredictor: slot_masks must be [B,N,P] or "
                    f"[B,N,H,W]; got shape {tuple(sm.shape)}."
                )
        # Cache reordered centroids if used. Use ``cache_A`` (hard permutation
        # in ST mode, soft Sinkhorn otherwise) so the cache matches the
        # forward semantics.
        if self.use_centroid_dist and curr_centroids is not None:
            self._prev_centroids = torch.einsum(
                "bij,bjk->bik", cache_A, curr_centroids.detach()
            )
        # Cache current step's flow for next step. forward_flow[t] represents
        # motion t -> t+1 and is exactly what the next step (which sees frame
        # t+1's slots) needs to warp ``_prev_masks`` (captured at t) into
        # t+1's coordinate system. ``flow`` may legitimately be None at the
        # final frame of a clip (no next-frame flow available); in that case
        # ``_prev_flow`` is reset to None so the IoU branch on the *next*
        # call will hard-raise (config asked for flow-IoU but no flow given)
        # rather than silently skip. For non-final frames flow must be
        # supplied — if ``use_flow_iou`` is on, hard-raise here.
        if flow is not None:
            self._prev_flow = flow.detach()
        else:
            if self.use_flow_iou:
                # Loud warning: flow-IoU is configured but the dataset is not
                # plumbing forward_flow at this step. We do NOT raise inside
                # forward (it would also fire on the final frame, which
                # legitimately has no flow), but we clear the cache so that
                # IoU on the next step is *not* silently skipped — it would
                # then raise via the explicit None check at the top of the
                # IoU branch on the following call.
                self._prev_flow = None
            else:
                self._prev_flow = None

        if existence_mask is not None:
            mask_reordered = torch.gather(existence_mask, 1, self._last_match_indices)
            return output, mask_reordered, None
        if return_weights:
            return output, A
        return output


class DepthStratifiedHungarianPredictor(HungarianPredictor):
    """Hungarian matcher restricted to operate **within depth strata**.

    Track III idea T3-01 (Depth-Stratified Hungarian). Crossing events cluster
    re-ID failures: two objects overlap in image space but have distinct
    depths. Restricting LSAP to a block-diagonal feasibility pattern — where
    slot i at t-1 can only match slot j at t if both fall in the same depth
    quantile bin — prevents swaps across distinct depth layers. Zero learned
    parameters.

    Mechanism (per §T3-01 of TRACK3_REID_IDEAS.md):
      - For each frame, compute per-slot mean depth as the soft-mask weighted
        average over the patch-resolution depth map.
      - Partition slots into ``depth_bins`` quantile bins *independently per
        frame* (``bin_mode="quantile"``). ``equal_width`` is also supported.
      - Build the standard cosine-distance cost matrix ``C[i,j]``, then add a
        large additive penalty (``penalty``, default 1e6) for every pair
        (i, j) whose prev-frame bin differs from curr-frame bin. The Hungarian
        solver still produces a valid bijection, but cross-bin pairings are
        only chosen when every in-bin alternative is also blocked.
      - ``depth_bins=1`` collapses the block-diagonal pattern to a single
        block, i.e. the penalty matrix is identically zero and the behaviour
        reduces exactly to :class:`HungarianPredictor`.

    Requirements:
      - Requires ``depth`` kwarg at every matching call (threaded via
        :class:`slotcontrast.modules.video.ScanOverTime` since the Phase 1.0
        depth plumbing patch). Missing depth raises ``RuntimeError`` — **no
        silent fallback**.
      - Requires current-frame ``masks`` and previous-frame ``prev_masks``
        (both [B, N, n_patches]) to compute per-slot mean depth.
      - Only implemented for ``pre_match=False`` and ``use_iterative=False``.
        Other modes raise in ``__init__``.

    Known concern (hysteresis default is off):
        Quantile bin boundaries are recomputed per frame, so a slot whose
        mean depth sits near a boundary may oscillate across adjacent frames
        → slot bin flicker → spurious penalty pattern changes → identity
        flicker. The ``hysteresis_band`` knob is reserved for a future
        sticky-bin fix (de-flickering); with the default ``0.0`` no
        stickiness is applied. **Monitor this at 10K steps** by inspecting
        per-slot bin histories; if flicker dominates, the idea degrades to a
        tuned method and the parameter-free claim no longer holds.
    """

    requires_depth = True

    def __init__(
        self,
        dim: int,
        depth_bins: int = 3,
        bin_mode: str = "quantile",
        hysteresis_band: float = 0.0,
        penalty: float = 1e6,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)

        if self.pre_match is not False:
            raise RuntimeError(
                "DepthStratifiedHungarianPredictor is only implemented for "
                f"pre_match=False (post-match mode); got pre_match={self.pre_match!r}."
            )
        if self.use_iterative:
            raise RuntimeError(
                "DepthStratifiedHungarianPredictor is not compatible with "
                "use_iterative=True; the iterative mutual-best matcher does "
                "not apply the depth-stratum penalty."
            )
        if bin_mode not in ("quantile", "equal_width"):
            raise ValueError(
                f"bin_mode must be 'quantile' or 'equal_width'; got {bin_mode!r}."
            )
        if depth_bins < 1:
            raise ValueError(f"depth_bins must be >= 1; got {depth_bins}.")
        if hysteresis_band != 0.0:
            # Parameter is accepted (per spec) but the de-flicker implementation
            # is intentionally deferred. Raising here enforces the "no silent
            # fallback" rule rather than ignoring the user's request.
            raise NotImplementedError(
                "hysteresis_band > 0 is reserved for a future sticky-bin "
                "implementation; leave it at 0.0 for the parameter-free default."
            )

        self.depth_bins = int(depth_bins)
        self.bin_mode = bin_mode
        self.hysteresis_band = float(hysteresis_band)
        self.penalty = float(penalty)

        # Cached depth for frame t-1. Populated every forward() call (whether
        # matching fires or not), cleared in reset().
        self._prev_depth: Optional[torch.Tensor] = None
        # Stash populated by forward() and consumed by _hungarian_match().
        self._dsh_penalty_matrix: Optional[torch.Tensor] = None

    def reset(self):
        super().reset()
        self._prev_depth = None
        self._dsh_penalty_matrix = None

    @staticmethod
    def _slot_mean_depth(masks: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Weighted mean depth per slot.

        Args:
            masks: [B, N, P] soft slot assignments over ``P`` patches.
            depth: [B, H, W] image-resolution depth map.

        Returns:
            mean_depth: [B, N] per-slot mean depth.
        """
        B, N, P = masks.shape
        p = int(round(P ** 0.5))
        if p * p != P:
            raise RuntimeError(
                f"DepthStratifiedHungarianPredictor expects square patch grids; "
                f"got P={P} which is not a perfect square."
            )
        # [B, 1, H, W] -> [B, 1, p, p] -> [B, P]
        depth_patch = F.adaptive_avg_pool2d(
            depth.unsqueeze(1).float(), (p, p)
        ).view(B, P)
        w_sum = masks.sum(dim=-1).clamp_min(1e-8)
        mean_d = (masks * depth_patch.unsqueeze(1)).sum(dim=-1) / w_sum
        return mean_d  # [B, N]

    def _assign_bins(self, mean_d: torch.Tensor) -> torch.Tensor:
        """Bin per-slot mean depth independently per batch element.

        Returns a [B, N] long tensor of bin indices in [0, depth_bins).
        depth_bins == 1 returns all-zeros (single block = no stratification).
        """
        B, N = mean_d.shape
        if self.depth_bins == 1:
            return torch.zeros(B, N, dtype=torch.long, device=mean_d.device)

        if self.bin_mode == "quantile":
            qs = torch.linspace(
                0.0, 1.0, self.depth_bins + 1, device=mean_d.device, dtype=mean_d.dtype
            )[1:-1]  # interior boundaries, [K-1]
            # torch.quantile(input, q, dim=1) with q of shape [K-1] returns
            # [K-1, B]; transpose to [B, K-1].
            edges = torch.quantile(mean_d, qs, dim=1).transpose(0, 1).contiguous()
        else:  # equal_width
            mn = mean_d.min(dim=1, keepdim=True)[0]
            mx = mean_d.max(dim=1, keepdim=True)[0]
            width = (mx - mn).clamp_min(1e-8) / self.depth_bins
            ks = torch.arange(
                1, self.depth_bins, device=mean_d.device, dtype=mean_d.dtype
            ).view(1, -1)
            edges = mn + ks * width  # [B, K-1]

        # torch.bucketize does not support batched boundaries; loop is cheap
        # since B is small (typical 8) and edges are short (K-1 = 2).
        bins = torch.zeros(B, N, dtype=torch.long, device=mean_d.device)
        for b in range(B):
            bins[b] = torch.bucketize(mean_d[b].contiguous(), edges[b].contiguous())
        return bins

    def _build_stratum_penalty(
        self, prev_bins: torch.Tensor, curr_bins: torch.Tensor
    ) -> torch.Tensor:
        """Return [B, N, N] penalty matrix: ``penalty`` for cross-bin pairs, 0 else."""
        diff = prev_bins.unsqueeze(-1) != curr_bins.unsqueeze(-2)  # [B, N, N]
        return diff.to(torch.float32) * self.penalty

    def forward(
        self,
        slots: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        centroids: Optional[torch.Tensor] = None,
        prev_centroids: Optional[torch.Tensor] = None,
        prev_prev_centroids: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        prev_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        depth = kwargs.get("depth", None)

        # Determine if this call will actually perform Hungarian matching.
        # Parent only matches in post-match mode when reference_slots is not
        # None; mirror that check here so we only require depth when we will
        # use it.
        reference_slots = prev_slots if prev_slots is not None else self._prev_slots
        will_match = reference_slots is not None  # pre_match is guaranteed False

        if will_match:
            if depth is None:
                raise RuntimeError(
                    "DepthStratifiedHungarianPredictor requires `depth` kwarg at "
                    "every matching call; none was provided. No fallback allowed."
                )
            if masks is None:
                raise RuntimeError(
                    "DepthStratifiedHungarianPredictor requires current-frame "
                    "`masks` kwarg ([B, N, P]) to compute per-slot mean depth."
                )
            if prev_masks is None:
                raise RuntimeError(
                    "DepthStratifiedHungarianPredictor requires previous-frame "
                    "`prev_masks` kwarg ([B, N, P]) to compute per-slot mean "
                    "depth at t-1."
                )
            if self._prev_depth is None:
                raise RuntimeError(
                    "DepthStratifiedHungarianPredictor: no cached depth from "
                    "t-1 at a matching call. Ensure ScanOverTime invokes the "
                    "predictor at t=0 with a non-None depth kwarg so the cache "
                    "can be populated. No fallback allowed."
                )

            prev_mean_d = self._slot_mean_depth(prev_masks, self._prev_depth)
            curr_mean_d = self._slot_mean_depth(masks, depth)
            prev_bins = self._assign_bins(prev_mean_d)
            curr_bins = self._assign_bins(curr_mean_d)
            self._dsh_penalty_matrix = self._build_stratum_penalty(prev_bins, curr_bins)
        else:
            self._dsh_penalty_matrix = None

        # Delegate all bookkeeping (pre/post-match branching, state updates,
        # existence_mask reordering, Kalman/velocity updates) to the parent.
        out = super().forward(
            slots,
            prev_slots=prev_slots,
            existence_mask=existence_mask,
            return_weights=return_weights,
            centroids=centroids,
            prev_centroids=prev_centroids,
            prev_prev_centroids=prev_prev_centroids,
            masks=masks,
            prev_masks=prev_masks,
            **kwargs,
        )

        # Cache current-frame depth for the next step; clear stash.
        if depth is not None:
            self._prev_depth = depth.detach()
        self._dsh_penalty_matrix = None

        return out

    def _hungarian_match(
        self,
        prev_slots: torch.Tensor,
        curr_slots: torch.Tensor,
        return_indices: bool = False,
        prev_centroids: Optional[torch.Tensor] = None,
        curr_centroids: Optional[torch.Tensor] = None,
        prev_prev_centroids: Optional[torch.Tensor] = None,
        prev_masks: Optional[torch.Tensor] = None,
        curr_masks: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Hungarian on the depth-stratified cost matrix.

        Replicates the parent's standard-Hungarian body but adds the penalty
        matrix that ``forward()`` stashed on ``self._dsh_penalty_matrix``.
        Does not support ``use_iterative=True`` (guarded in ``__init__``).
        """
        from scipy.optimize import linear_sum_assignment

        B, N, D = curr_slots.shape
        device = curr_slots.device

        if self.use_hybrid_cost:
            cost_matrix = self._compute_hybrid_cost(
                prev_slots, curr_slots, prev_centroids, curr_centroids,
                prev_prev_centroids, prev_masks, curr_masks,
            )
        else:
            if self.similarity == "cosine":
                prev_norm = F.normalize(prev_slots, dim=-1)
                curr_norm = F.normalize(curr_slots, dim=-1)
                sim_matrix = torch.bmm(prev_norm, curr_norm.transpose(1, 2))
                cost_matrix = 1 - sim_matrix
            else:
                diff = prev_slots.unsqueeze(2) - curr_slots.unsqueeze(1)
                cost_matrix = diff.norm(dim=-1)

        if self._dsh_penalty_matrix is not None:
            cost_matrix = cost_matrix + self._dsh_penalty_matrix.to(cost_matrix.dtype)

        reordered_list = []
        indices_list = []
        for b in range(B):
            cost_np = cost_matrix[b].detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            reordered = curr_slots[b, col_ind]
            reordered_list.append(reordered)
            indices_list.append(torch.tensor(col_ind, device=device, dtype=torch.long))

        reordered = torch.stack(reordered_list, dim=0)
        if return_indices:
            indices = torch.stack(indices_list, dim=0)
            return reordered, indices
        return reordered


class HungarianMemoryMatcher(nn.Module):
    """Memory-based Hungarian matcher with persistent track IDs (no deletion).
    
    Three cases:
    1. MATCHED: Update registry slot with EMA
    2. UNMATCHED candidate: Register as new object in first empty slot  
    3. OCCLUDED registry: Retain existing feature unchanged
    
    Compatible with HungarianPredictor V1 interface.
    
    Note: Registry is per-batch-element. Each batch element (independent video)
    maintains its own registry across time steps within a video chunk.
    """
    
    def __init__(
        self,
        dim: int,
        max_slots: int = 15,
        match_threshold: float = 0.5,  # Cosine distance threshold for valid match
        ema_decay: float = 0.9,  # EMA weight: new = decay * old + (1-decay) * current
        similarity: str = "cosine",
        pre_match: bool = False,  # V1 compatibility
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.max_slots = max_slots
        self.match_threshold = match_threshold
        self.ema_decay = ema_decay
        self.similarity = similarity
        self.pre_match = pre_match
        
        # Per-batch-element registries (initialized in forward)
        # Shape: [B, max_slots, dim] and [B, max_slots]
        self._registry_features: Optional[torch.Tensor] = None
        self._registry_occupied: Optional[torch.Tensor] = None
        self._batch_size: int = 0
        
        # V1 compatibility
        self._prev_slots: Optional[torch.Tensor] = None
        self._last_match_indices: Optional[torch.Tensor] = None
    
    def reset(self):
        """Reset for new video sequence (called at start of each chunk)."""
        self._registry_features = None
        self._registry_occupied = None
        self._batch_size = 0
        self._prev_slots = None
        self._last_match_indices = None
    
    def _init_registry(self, batch_size: int, device: torch.device):
        """Initialize per-batch-element registries."""
        self._registry_features = torch.zeros(batch_size, self.max_slots, self.dim, device=device)
        self._registry_occupied = torch.zeros(batch_size, self.max_slots, dtype=torch.bool, device=device)
        self._batch_size = batch_size
    
    def _hungarian_match(self, prev_slots: torch.Tensor, curr_slots: torch.Tensor, return_indices: bool = False):
        """V1 compatibility: standard Hungarian matching without memory."""
        from scipy.optimize import linear_sum_assignment
        B, N, D = curr_slots.shape
        device = curr_slots.device
        
        if self.similarity == "cosine":
            prev_norm = F.normalize(prev_slots, dim=-1)
            curr_norm = F.normalize(curr_slots, dim=-1)
            sim_matrix = torch.bmm(prev_norm, curr_norm.transpose(1, 2))
            cost_matrix = 1 - sim_matrix
        else:
            diff = prev_slots.unsqueeze(2) - curr_slots.unsqueeze(1)
            cost_matrix = diff.norm(dim=-1)
        
        reordered_list, indices_list = [], []
        for b in range(B):
            cost_np = cost_matrix[b].detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            reordered_list.append(curr_slots[b, col_ind])
            indices_list.append(torch.tensor(col_ind, device=device, dtype=torch.long))
        
        reordered = torch.stack(reordered_list, dim=0)
        if return_indices:
            return reordered, torch.stack(indices_list, dim=0)
        return reordered
    
    def _match_and_update_single(self, b: int, candidates: torch.Tensor) -> tuple:
        """Match candidates to registry for a single batch element.
        
        Args:
            b: batch index
            candidates: [n_valid, D] valid slots for this batch element
            
        Returns:
            out_slots: [max_slots, D] reordered slots with gradients
            out_mask: [max_slots] existence mask
            indices: [n_candidates] mapping from candidate idx to output slot idx
        """
        from scipy.optimize import linear_sum_assignment
        device = candidates.device
        n_candidates = candidates.shape[0]
        
        # Get this batch element's registry
        reg_features = self._registry_features[b]  # [max_slots, D]
        reg_occupied = self._registry_occupied[b]  # [max_slots]
        
        occupied_idx = reg_occupied.nonzero(as_tuple=True)[0]
        n_occupied = len(occupied_idx)
        
        # Output: reordered current slots (with gradients) + mask + indices
        out_slots = torch.zeros(self.max_slots, candidates.shape[-1], device=device)
        out_mask = torch.zeros(self.max_slots, device=device)
        cand_to_out = torch.full((n_candidates,), -1, dtype=torch.long, device=device)
        
        if n_occupied == 0:
            # First frame: register all candidates, assign to slots 0..n-1
            n_new = min(n_candidates, self.max_slots)
            for i in range(n_new):
                self._registry_features[b, i] = candidates[i].detach()
                self._registry_occupied[b, i] = True
                out_slots[i] = candidates[i]  # Output has gradients
                out_mask[i] = 1.0
                cand_to_out[i] = i
            return out_slots, out_mask, cand_to_out
        
        # Compute cost matrix: candidates vs occupied registry slots
        cand_norm = F.normalize(candidates.detach(), dim=-1, eps=1e-8)
        reg_norm = F.normalize(reg_features[occupied_idx], dim=-1, eps=1e-8)
        cost_matrix = 1 - (cand_norm @ reg_norm.t())
        cost_matrix = torch.nan_to_num(cost_matrix, nan=2.0)  # NaN → max cost
        
        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
        
        matched_registry = set()  # Registry slots with ACCEPTED matches
        matched_candidates = set()  # Candidates with ACCEPTED matches
        
        # First pass: separate accepted vs rejected matches based on threshold
        for cand_idx, occ_idx in zip(row_ind, col_ind):
            cost = cost_matrix[cand_idx, occ_idx].item()
            reg_idx = occupied_idx[occ_idx].item()
            
            if cost <= self.match_threshold:
                # ACCEPTED match: output to registry slot and update with EMA
                matched_registry.add(reg_idx)
                matched_candidates.add(cand_idx)
                out_slots[reg_idx] = candidates[cand_idx]  # Original with gradients
                out_mask[reg_idx] = 1.0
                cand_to_out[cand_idx] = reg_idx
                
                # Update registry with EMA
                self._registry_features[b, reg_idx] = (
                    self.ema_decay * self._registry_features[b, reg_idx] +
                    (1 - self.ema_decay) * candidates[cand_idx].detach()
                )
            # REJECTED match (cost > threshold): candidate treated as new object below
        
        # UNMATCHED or REJECTED candidates: assign to new registry slots
        for cand_idx in range(n_candidates):
            if cand_idx not in matched_candidates:
                empty_slots = (~self._registry_occupied[b]).nonzero(as_tuple=True)[0]
                if len(empty_slots) > 0:
                    new_idx = empty_slots[0].item()
                    self._registry_features[b, new_idx] = candidates[cand_idx].detach()
                    self._registry_occupied[b, new_idx] = True
                    out_slots[new_idx] = candidates[cand_idx]  # Original with gradients
                    out_mask[new_idx] = 1.0
                    cand_to_out[cand_idx] = new_idx
        
        # OCCLUDED: registry slots not matched (or match was rejected) this frame
        # Keep mask=0 (empty for this frame), registry retains features for future
        for idx in occupied_idx:
            if idx.item() not in matched_registry:
                out_mask[idx] = 0.0  # Not visible this frame
        
        return out_slots, out_mask, cand_to_out
    
    def forward(
        self,
        slots: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        """
        Args:
            slots: [B, K, D] current frame slots
            prev_slots: [B, K, D] previous slots (V1 compatibility, ignored if using memory)
            existence_mask: [B, K] which slots are valid (1=valid, 0=empty)
            return_weights: V1 compatibility
        
        Returns (memory mode):
            out_slots: [B, max_slots, D] ordered by persistent global ID
            out_mask: [B, max_slots] - 1.0=matched, 0.0=empty/occluded
        Returns (V1 mode when existence_mask is None):
            reordered_slots: [B, K, D]
        """
        B, K, D = slots.shape
        device = slots.device
        
        # V1 compatibility: if no existence_mask, use standard Hungarian
        if existence_mask is None:
            reference = prev_slots if prev_slots is not None else self._prev_slots
            if reference is None:
                self._prev_slots = slots.detach()
                self._last_match_indices = None
                return (slots, None) if return_weights else slots
            
            reordered, indices = self._hungarian_match(reference, slots, return_indices=True)
            self._prev_slots = reordered.detach()
            self._last_match_indices = indices
            return (reordered, None) if return_weights else reordered
        
        # Memory mode: initialize per-batch registries if needed
        if self._registry_features is None or self._batch_size != B:
            self._init_registry(B, device)
        
        # Process each batch element with its own registry
        out_slots = torch.zeros(B, self.max_slots, D, device=device)
        out_mask = torch.zeros(B, self.max_slots, device=device)
        # Track match indices: [B, max_slots] where value is output slot idx
        match_indices = torch.arange(self.max_slots, device=device).unsqueeze(0).expand(B, -1).clone()
        
        for b in range(B):
            valid_mask = existence_mask[b].bool()
            valid_slots = slots[b, valid_mask]
            
            if valid_slots.shape[0] == 0:
                # No valid slots: return zeros with empty mask
                continue
            
            out_slots[b], out_mask[b], cand_to_out = self._match_and_update_single(b, valid_slots)
            # Build reverse mapping: for each output slot, which candidate went there
            for cand_idx, out_idx in enumerate(cand_to_out.tolist()):
                if out_idx >= 0:
                    match_indices[b, out_idx] = cand_idx
        
        self._last_match_indices = match_indices
        
        if return_weights:
            return out_slots, out_mask, None
        return out_slots, out_mask
    
    def match_to_reference(self, slots: torch.Tensor) -> torch.Tensor:
        """V1 interface for pre-matching mode."""
        out, _ = self.forward(slots, existence_mask=None)
        return out if isinstance(out, torch.Tensor) else out[0]
    
    def get_last_match_indices(self) -> Optional[torch.Tensor]:
        """V1 interface."""
        return self._last_match_indices


class MAPWithRejectPredictor(nn.Module):
    """Witness algorithm for the Open-Set Identity Capacity theorem (GCv2).

    Implements the MAP-with-reject classifier from Theorem 3 as an online
    slot predictor. Drop-in compatible with HungarianPredictor.

    Three temporal decisions, all derived from the capacity law:
      CONTINUE: slot matches an active previous slot (argmin cost < threshold)
      RE-ENTER: slot matches a dormant registry entry (same test, older centroid)
      BIRTH:    slot matches nothing (all costs > threshold) → new identity

    Plus the inverse:
      DEATH: an active slot from t-1 has no match in t → moves to dormant registry

    The reject threshold is derived from the theorem: at distance > Δ_min/2
    the MAP classifier errors with probability > 0.5, so rejecting is optimal.
    In practice we use a configurable cosine-distance threshold.

    Zero learnable parameters. Preserves the correspondence thesis.
    """

    def __init__(
        self,
        dim: int,
        similarity: str = "cosine",
        # Dormant registry settings
        max_dormant_age: int = 10,
        reject_threshold: float = 0.5,  # cosine distance above which → birth/reject
        age_penalty: float = 0.05,      # cost penalty per frame of dormancy
        # Interface compatibility
        pre_match: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.similarity = similarity
        self.max_dormant_age = max_dormant_age
        self.reject_threshold = reject_threshold
        self.age_penalty = age_penalty
        self.pre_match = pre_match

        # Sentinel so video.py dispatches existence_mask correctly
        # (video.py checks hasattr(predictor, '_hungarian_match'))
        self._hungarian_match = None  # not used, just for dispatch

        # State (per-video, reset between videos)
        self._prev_slots: Optional[torch.Tensor] = None       # [B, N, D]
        self._dormant: Optional[List[List[Dict]]] = None       # per-batch list of dormant entries
        self._last_match_indices: Optional[torch.Tensor] = None
        self._next_id: Optional[List[int]] = None              # per-batch next fresh ID counter
        # Per-slot LSAP margin (chosen-cost minus second-best cost on the same row).
        # Detached scalar diagnostic; consumed by HybridPredictor's gating step.
        self._last_margin_per_slot: Optional[torch.Tensor] = None  # [B, N]
        # Birth-overflow flag (set inside forward); HybridPredictor checks this.
        self._last_births_overflow: bool = False
        # Persisted existence mask after the most recent forward (used by HybridPredictor).
        self._last_existence_mask: Optional[torch.Tensor] = None

    @property
    def last_margin_per_slot(self) -> Optional[torch.Tensor]:
        """Per-slot LSAP cost margin from the most recent forward call.

        Returns ``None`` on the first frame (no LSAP solved). Otherwise returns a
        detached ``[B, N]`` tensor where each entry is ``second_best - chosen``
        (larger ⇒ more confident assignment; smaller ⇒ ambiguous match).
        """
        return self._last_margin_per_slot

    def reset(self):
        """Reset for new video sequence."""
        self._prev_slots = None
        self._dormant = None
        self._last_match_indices = None
        self._next_id = None
        self._last_margin_per_slot = None
        self._last_births_overflow = False
        self._last_existence_mask = None

    def forward(
        self,
        slots: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            slots: [B, N, D] current frame slots (from slot attention)
            prev_slots: [B, N, D] optional override for previous slots
            existence_mask: [B, N] which slots are valid

        Returns:
            Reordered slots to maintain temporal identity [B, N, D]
        """
        from scipy.optimize import linear_sum_assignment

        B, N, D = slots.shape
        device = slots.device
        reference = prev_slots if prev_slots is not None else self._prev_slots

        # First frame: initialize state
        if reference is None:
            self._prev_slots = slots.detach()
            self._dormant = [[] for _ in range(B)]
            self._next_id = [N for _ in range(B)]  # IDs 0..N-1 taken by first frame
            self._last_match_indices = None
            # No LSAP solved on the first frame: use a large positive margin so a
            # downstream gate (sigmoid((tau - margin)/T)) reads ~0 (non-ambiguous).
            self._last_margin_per_slot = torch.full(
                (B, N), float("inf"), device=device, dtype=slots.dtype
            )
            self._last_births_overflow = False
            self._last_existence_mask = existence_mask
            if existence_mask is not None:
                return slots, existence_mask, None
            return slots if not return_weights else (slots, None)

        # Normalize for cosine distance
        ref_norm = F.normalize(reference, dim=-1)      # [B, N, D]
        cur_norm = F.normalize(slots, dim=-1)           # [B, N, D]

        reordered_list = []
        indices_list = []
        margin_list = []
        # Reset birth-overflow flag for this forward; set to True if any batch
        # element triggers the warning path below.
        self._last_births_overflow = False

        for b in range(B):
            # Build augmented cost matrix: [N_cur rows] × [N_active + N_dormant cols]
            # Active columns: previous frame's N slots
            # Dormant columns: registry entries (aged, penalty-adjusted)
            active_feats = ref_norm[b]   # [N, D]
            dormant_entries = self._dormant[b]

            n_active = N
            n_dormant = len(dormant_entries)
            n_cols = n_active + n_dormant

            # Current slot features
            cur_feats = cur_norm[b]  # [N, D]

            # Cost matrix: cosine distance
            # Active block: [N, N_active]
            active_cost = 1.0 - (cur_feats @ active_feats.T)  # [N, N]

            # If existence_mask provided, mask invalid slots to high cost
            # so they cannot win matches or corrupt registry state
            if existence_mask is not None:
                cur_invalid = ~existence_mask[b].bool()  # [N] True = invalid
                prev_invalid = cur_invalid  # same mask shape for prev (fixed K)
                active_cost[cur_invalid, :] = 1e6   # invalid current → can't match
                active_cost[:, prev_invalid] = 1e6  # invalid prev → can't be matched

            if n_dormant > 0:
                dormant_feats = torch.stack(
                    [d["feature"] for d in dormant_entries]
                ).to(device)  # [n_dormant, D]
                dormant_feats = F.normalize(dormant_feats, dim=-1)
                dormant_cost = 1.0 - (cur_feats @ dormant_feats.T)  # [N, n_dormant]
                # Age penalty: older dormant entries cost more to re-activate
                ages = torch.tensor(
                    [d["age"] for d in dormant_entries],
                    device=device, dtype=torch.float32
                )
                dormant_cost = dormant_cost + self.age_penalty * ages.unsqueeze(0)
                # Full cost matrix
                cost = torch.cat([active_cost, dormant_cost], dim=1)  # [N, n_cols]
            else:
                cost = active_cost  # [N, N]

            # Pad to square if needed (more cols than rows or vice versa)
            n_rows = N
            if n_cols > n_rows:
                # More candidates than current slots: pad rows with high cost (dummy slots)
                pad = torch.full((n_cols - n_rows, n_cols), 1e6, device=device)
                cost_sq = torch.cat([cost, pad], dim=0)
            elif n_rows > n_cols:
                # More current slots than candidates: pad cols (will become births)
                pad = torch.full((n_rows, n_rows - n_cols), self.reject_threshold + 0.01,
                                  device=device)
                cost_sq = torch.cat([cost, pad], dim=1)
            else:
                cost_sq = cost

            # Solve LSAP
            cost_np = cost_sq.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            # --- Per-slot ambiguity margin (consumed by HybridPredictor) ---
            # Margin = (second-best column cost) - (chosen column cost) on the
            # same row. Larger ⇒ confident assignment. We iterate over the first
            # N rows only (real current slots; padded rows are not real slots).
            # Computed on the detached cost (no gradient into LSAP).
            cost_sq_det = cost_sq.detach()
            # Build a per-row index lookup so that we can subtract the chosen
            # column's cost without a Python loop.
            #   chosen = cost_sq_det[row_ind[:N], col_ind[:N]]
            #   masked = cost_sq_det[row_ind[:N]].masked_fill(one_hot, +inf)
            #   second = masked.min(dim=-1)
            if N > 0:
                row_idx_t = torch.as_tensor(row_ind[:N], device=device, dtype=torch.long)
                col_idx_t = torch.as_tensor(col_ind[:N], device=device, dtype=torch.long)
                row_costs = cost_sq_det.index_select(0, row_idx_t)  # [N, n_cols_sq]
                chosen_costs = row_costs.gather(1, col_idx_t.unsqueeze(-1)).squeeze(-1)  # [N]
                inf_src = torch.full_like(col_idx_t.unsqueeze(-1), 0, dtype=row_costs.dtype)
                inf_src.fill_(float("inf"))
                masked = row_costs.scatter(1, col_idx_t.unsqueeze(-1), inf_src)
                if masked.shape[1] > 1:
                    second_costs, _ = masked.min(dim=-1)
                else:
                    # Only one column to choose from: margin is undefined → +inf.
                    second_costs = torch.full_like(chosen_costs, float("inf"))
                # Place the margin at the OUTPUT slot position (col index when
                # the row continues an active prev slot), so the value aligns
                # with the reordered output. For births / re-enters we set the
                # margin at the eventual output position later in pass 2.
                row_margins = second_costs - chosen_costs  # [N]; >=0 typically
            else:
                row_margins = torch.empty(0, device=device, dtype=slots.dtype)

            # Interpret the assignment.
            # Contract: output slot at position i should carry the identity that
            # was at position i in the previous frame.
            # LSAP gives (row=current_slot, col=prev_identity_or_dormant).
            #
            # Two passes:
            #   Pass 1: place CONTINUE and RE-ENTER slots at their target positions
            #   Pass 2: place BIRTH slots at unclaimed positions

            output = torch.zeros_like(slots[b])  # [N, D]
            source_for_identity = torch.full((N,), -1, device=device, dtype=torch.long)
            claimed_positions = set()  # output positions already taken
            birth_rows = []            # current rows that need birth assignment
            new_dormant = []
            matched_active = set()
            matched_dormant = set()
            # Track which row produced each output position so we can write the
            # per-row LSAP margin at the corresponding output index.
            out_pos_to_row: List[int] = [-1] * N

            # Pass 1a: CONTINUE (active matches have priority over dormant)
            reenter_candidates = []  # (row, dormant_idx) pairs for pass 1b
            for r, c in zip(row_ind[:N], col_ind[:N]):
                actual_cost = cost[r, c].item() if c < n_cols else self.reject_threshold + 1

                if actual_cost > self.reject_threshold:
                    birth_rows.append(int(r))
                elif c < n_active:
                    # CONTINUE: current slot r → output position c
                    output[c] = slots[b, r]
                    source_for_identity[c] = r
                    claimed_positions.add(c)
                    matched_active.add(c)
                    out_pos_to_row[c] = int(r)
                else:
                    # Defer RE-ENTER to pass 1b (after all CONTINUEs claim positions)
                    d_idx = c - n_active
                    reenter_candidates.append((int(r), d_idx))
                    # NOTE: do NOT add to matched_dormant yet — collision may demote to birth

            # Pass 1b: RE-ENTER (only if orig_idx is still unclaimed)
            for row_idx, d_idx in reenter_candidates:
                orig_idx = dormant_entries[d_idx].get("orig_idx", -1)
                if 0 <= orig_idx < N and orig_idx not in claimed_positions:
                    output[orig_idx] = slots[b, row_idx]
                    source_for_identity[orig_idx] = row_idx
                    claimed_positions.add(orig_idx)
                    matched_dormant.add(d_idx)  # only on successful re-entry
                    out_pos_to_row[orig_idx] = int(row_idx)
                else:
                    # Collision or invalid → demote to birth; dormant entry stays in registry
                    birth_rows.append(row_idx)

            # Pass 2: BIRTH — place at unclaimed output positions
            unclaimed = sorted(set(range(N)) - claimed_positions)
            for row_idx, out_pos in zip(birth_rows, unclaimed):
                output[out_pos] = slots[b, row_idx]
                source_for_identity[out_pos] = row_idx
                claimed_positions.add(out_pos)
                out_pos_to_row[out_pos] = int(row_idx)
            # If more births than unclaimed (shouldn't happen with correct padding),
            # remaining births are silently dropped (logged for debugging)
            if len(birth_rows) > len(unclaimed):
                import warnings
                warnings.warn(
                    f"MAPWithRejectPredictor: {len(birth_rows)} births but only "
                    f"{len(unclaimed)} unclaimed positions; dropping "
                    f"{len(birth_rows) - len(unclaimed)} births"
                )
                self._last_births_overflow = True

            # DEATH: active slots from t-1 with no match → push to dormant
            # Store the original slot index so RE-ENTER can restore it later
            for j in range(n_active):
                if j not in matched_active:
                    new_dormant.append({
                        "feature": reference[b, j].detach().cpu(),
                        "age": 0,
                        "orig_idx": j,  # <-- FIX: track original identity index
                    })

            # Update dormant: keep unmatched entries, age all, evict old
            for d_idx, entry in enumerate(dormant_entries):
                if d_idx not in matched_dormant:
                    entry["age"] += 1
                    if entry["age"] <= self.max_dormant_age:
                        new_dormant.append(entry)

            self._dormant[b] = new_dormant
            reordered_list.append(output)
            indices_list.append(source_for_identity)

            # Project the per-row margins into per-output-slot margins.
            # Output slots that received no row (none of the assignments
            # populated them) get +inf, which a downstream gate reads as
            # "fully confident / non-ambiguous".
            if N > 0:
                out_margins_b = torch.full(
                    (N,), float("inf"), device=device, dtype=slots.dtype
                )
                for out_pos, r in enumerate(out_pos_to_row):
                    if 0 <= r < N:
                        out_margins_b[out_pos] = row_margins[r]
                margin_list.append(out_margins_b)
            else:
                margin_list.append(row_margins)

        reordered = torch.stack(reordered_list, dim=0)  # [B, N, D]
        self._prev_slots = reordered.detach()
        self._last_match_indices = torch.stack(indices_list, dim=0)  # [B, N]
        # Stack and store the per-output-slot margin diagnostic. Detached so it
        # cannot leak gradient into LSAP / cost matrix construction.
        if len(margin_list) > 0:
            self._last_margin_per_slot = torch.stack(margin_list, dim=0).detach()
        else:
            self._last_margin_per_slot = torch.empty(
                (B, 0), device=device, dtype=slots.dtype
            )

        if existence_mask is not None:
            reordered_mask = torch.stack(
                [existence_mask[b, indices_list[b]] for b in range(B)], dim=0
            )
            self._last_existence_mask = reordered_mask
            return reordered, reordered_mask, None
        self._last_existence_mask = None
        if return_weights:
            return reordered, None
        return reordered

    def get_last_match_indices(self) -> Optional[torch.Tensor]:
        return self._last_match_indices

    def match_to_reference(self, slots: torch.Tensor) -> torch.Tensor:
        """V1 interface for pre-matching mode."""
        out = self.forward(slots)
        return out if isinstance(out, torch.Tensor) else out[0]


class HybridPredictor(nn.Module):
    """Ambiguity-gated wrapper around :class:`MAPWithRejectPredictor` (GCv2 v9).

    .. note::
       **QUARANTINED — not on the active TPAMI rescue path (Round-32 audit).**
       The active rescue protocol uses :class:`SoftIdentityPredictor` (or the
       baseline TransformerEncoder predictor in the v2 grid). HybridPredictor
       is a Phase 1 prototype carried over from the GCv2 spec; the zero-return
       branches in :meth:`warp_patches_by_flow` are explicit "shape-correct
       stub" returns for the unfinished flow-warp implementation, not silent
       failure handlers. No app config in v2/, phase_a/, phase_b/, or F.4
       instantiates this class. Use :class:`SoftIdentityPredictor` for the
       TPAMI rescue protocol.

    Per the HybridPredictor + Cluster-Merge spec (R0.4), this module is a
    drop-in for :class:`LatentProcessor`'s predictor contract. It calls the
    inner MAP-with-reject predictor for the discrete identity assignment, then
    mixes in *learnable* corrections only on slots whose LSAP cost margin says
    the assignment is ambiguous. Three correction sources are available, all
    individually toggleable for ablation:

      * ambiguity-gated cross-attention over current-frame DINOv2 patches,
      * a frozen optical-flow branch that projects warped patch evidence,
      * a count head whose probabilities arbitrate cluster-merges.

    The wrapper never modifies the inner MAP module in-place; it only reads its
    diagnostics (``last_margin_per_slot``, ``_last_births_overflow``) and
    overwrites the cached ``_prev_slots`` *after* its own corrections so the
    next-frame LSAP sees the corrected state (§4 of the spec).

    Differentiability (§3 of the spec):
      * LSAP / margin / merge: forward-only, detached.
      * Cross-attention, motion projection, count head, gate parameters:
        learnable.
      * Sparsity penalty ``mean(z)`` is computable by the caller from the
        returned auxiliary tensor.
    """

    def __init__(
        self,
        dim: int,
        # --- MAPWithReject pass-through ---
        reject_threshold: float = 0.5,
        age_penalty: float = 0.05,
        max_dormant_age: int = 10,
        similarity: str = "cosine",
        pre_match: bool = False,
        # --- Gate parameterization ---
        gate_mode: str = "calibrated",
        tau_gate: float = 0.2,
        T_gate_init: float = 0.05,
        lambda_sparse: float = 1e-3,
        # --- Branch toggles ---
        use_xattn: bool = True,
        use_motion: bool = True,
        motion_standalone: bool = False,
        motion_source: str = "raft",
        # --- Cross-attention hyper-params ---
        num_heads: int = 8,
        ffn_dim: int = 2048,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        # --- Count head ---
        use_count_head: bool = True,
        count_hidden: int = 256,
        # --- Cluster-merge ---
        use_cluster_merge: bool = True,
        merge_cosine_threshold: float = 0.90,
        merge_count_winner_min: float = 0.7,
        merge_count_loser_max: float = 0.3,
        merge_cooldown_frames: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if gate_mode not in ("calibrated", "learned"):
            raise ValueError(
                f"gate_mode must be 'calibrated' or 'learned', got {gate_mode!r}"
            )
        if T_gate_init <= 0:
            raise ValueError(f"T_gate_init must be positive, got {T_gate_init}")

        self.dim = dim
        self.gate_mode = gate_mode
        self.lambda_sparse = float(lambda_sparse)
        self.use_xattn = bool(use_xattn)
        self.use_motion = bool(use_motion)
        self.motion_standalone = bool(motion_standalone)
        self.motion_source = motion_source
        self.use_count_head = bool(use_count_head)
        self.use_cluster_merge = bool(use_cluster_merge)
        self.merge_cosine_threshold = float(merge_cosine_threshold)
        self.merge_count_winner_min = float(merge_count_winner_min)
        self.merge_count_loser_max = float(merge_count_loser_max)
        self.merge_cooldown_frames = int(merge_cooldown_frames)
        self.num_heads = int(num_heads)

        # --- Inner MAP predictor (NOT modified in-place) ---
        self.map = MAPWithRejectPredictor(
            dim=dim,
            similarity=similarity,
            max_dormant_age=max_dormant_age,
            reject_threshold=reject_threshold,
            age_penalty=age_penalty,
            pre_match=pre_match,
        )

        # Sentinel so LatentProcessor's dispatch routes us through the
        # `is_hungarian` branch (we accept ``existence_mask`` and return the
        # ``(slots, mask, aux)`` triple it expects).
        self._hungarian_match = None
        # Match the inner module's flag for callers that read ``pre_match``
        # directly off the predictor.
        self.pre_match = pre_match

        # --- Gate parameters ---
        if gate_mode == "calibrated":
            self.register_buffer("tau_gate", torch.tensor(float(tau_gate)))
            self.register_buffer("T_gate", torch.tensor(float(T_gate_init)))
            # Placeholder so ``T_gate_raw`` always exists for state-dict round
            # trips; not used for the calibrated path.
            self.register_buffer(
                "T_gate_raw",
                torch.tensor(math.log(math.expm1(float(T_gate_init)))),
            )
        else:
            # Learned τ as a free Parameter; learned T via softplus param so it
            # stays strictly positive without clamping.
            self.tau_gate = nn.Parameter(torch.tensor(float(tau_gate)))
            self.T_gate_raw = nn.Parameter(
                torch.tensor(math.log(math.expm1(float(T_gate_init))))
            )

        # --- Cross-attention block (only built if use_xattn or motion_cond) ---
        if self.use_xattn or (self.use_motion and not self.motion_standalone):
            self.q_norm = nn.LayerNorm(dim)
            self.kv_norm = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=self.num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            self.attn_post_norm = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(ffn_dim, dim),
                nn.Dropout(ffn_dropout),
            )
            self.ffn_norm = nn.LayerNorm(dim)
        else:
            self.q_norm = None
            self.kv_norm = None
            self.cross_attn = None
            self.attn_post_norm = None
            self.ffn = None
            self.ffn_norm = None

        # --- Motion branch ---
        # A single learned linear layer projects warped DINOv2 patches into the
        # slot space; per spec §8 this is the ONLY learned param touching the
        # frozen flow stream.
        if self.use_motion:
            self.motion_projection = nn.Linear(dim, dim)
            self.motion_norm = nn.LayerNorm(dim)
        else:
            self.motion_projection = None
            self.motion_norm = None

        # --- Count head (small MLP → scalar logit per slot) ---
        if self.use_count_head:
            self.count_mlp = nn.Sequential(
                nn.Linear(dim, count_hidden),
                nn.GELU(),
                nn.Linear(count_hidden, 1),
            )
        else:
            self.count_mlp = None

        # --- Cluster-merge cooldown registry: dict[pair_id] -> frames left ---
        # Persisted across forward calls; cleared by ``reset()``.
        self._cooldown_registry: Dict[Tuple[int, int, int], int] = {}

    # ------------------------------------------------------------------ utils
    @property
    def T_gate_positive(self) -> torch.Tensor:
        """Strictly-positive gate temperature (softplus-parameterized when learned)."""
        if self.gate_mode == "calibrated":
            return self.T_gate
        return F.softplus(self.T_gate_raw) + 1e-4

    def reset(self) -> None:
        """Reset per-video state. Forwarded to the inner MAP module."""
        self.map.reset()
        self._cooldown_registry = {}

    def get_last_match_indices(self) -> Optional[torch.Tensor]:
        return self.map.get_last_match_indices()

    def match_to_reference(self, slots: torch.Tensor) -> torch.Tensor:
        return self.map.match_to_reference(slots)

    @staticmethod
    def _pair_id(b: int, i: int, j: int) -> Tuple[int, int, int]:
        """Stable, order-independent identifier for a (loser, winner) pair."""
        lo, hi = (i, j) if i < j else (j, i)
        return (int(b), int(lo), int(hi))

    # --------------------------------------------------------- motion helper
    def warp_patches_by_flow(
        self,
        flow: torch.Tensor,
        slots: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        slot_centroids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute a per-slot warped patch evidence vector (Phase 1 stub).

        Spec §4.1 calls for: bilinearly sample ``flow`` at each slot's support
        centroid, then mean-pool a 3x3 neighborhood of DINOv2 patches at the
        warped location. Phase 1 ships a minimal but shape-correct stub so the
        rest of the pipeline can be exercised end-to-end. A full implementation
        will land alongside the patch-centroid plumbing in ``video.py``.

        Args:
            flow: ``[B, H, W, 2]`` forward flow (t-1 → t). Detached.
            slots: ``[B, N, D]`` current slot features (used for shape only).
            features: ``[B, P, D]`` current-frame patch features (optional, used
                when patch-level pooling becomes available).
            slot_centroids: ``[B, N, 2]`` slot centroid in normalized image
                coordinates ``(y, x) ∈ [-1, 1]`` (optional).

        Returns:
            ``[B, N, D]`` warped patch evidence, detached so no gradient enters
            the flow model.
        """
        B, N, D = slots.shape
        device = slots.device
        dtype = slots.dtype

        # Defensive: if there is no flow we return zeros; the caller guards on
        # this case but the helper should be robust on its own.
        if flow is None:
            return torch.zeros(B, N, D, device=device, dtype=dtype)

        flow_d = flow.detach()
        if flow_d.shape[0] != B:
            # Mismatched batch: cannot warp — return zeros so the residual
            # contribution vanishes.
            return torch.zeros(B, N, D, device=device, dtype=dtype)

        # Phase 1 stub: when features and centroids are available, sample a
        # single patch per slot at the warped centroid and project later. When
        # they're not, return a zero contribution so the gated residual is a
        # no-op (still satisfies the [B, N, D] contract).
        if features is None or slot_centroids is None:
            return torch.zeros(B, N, D, device=device, dtype=dtype)

        # --- Minimal "sample one warped patch per slot" implementation ---
        H, W = flow_d.shape[1], flow_d.shape[2]
        # Sample flow at slot centroids (normalized [-1, 1]) using
        # grid_sample. ``grid_sample`` expects (B, C, H, W) input and (B, H_q,
        # W_q, 2) grid where the LAST axis is (x, y).
        flow_chw = flow_d.permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]
        grid_xy = slot_centroids.detach().clone()  # [B, N, 2] (y, x) → swap
        grid_xy = grid_xy[..., [1, 0]]            # now (x, y)
        grid_q = grid_xy.unsqueeze(2)             # [B, N, 1, 2]
        sampled_flow = F.grid_sample(
            flow_chw, grid_q,
            mode="bilinear", padding_mode="border", align_corners=True,
        ).squeeze(-1).permute(0, 2, 1)            # [B, N, 2]

        # Warp centroid by flow (flow is in pixel units; convert to normalized
        # [-1, 1] by dividing by half image extent). We assume the caller
        # already passes a centroid in [-1, 1].
        scale = torch.tensor(
            [2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)],
            device=device, dtype=dtype,
        )
        warped_xy = grid_xy + sampled_flow * scale  # [B, N, 2]
        warped_grid = warped_xy.unsqueeze(2)        # [B, N, 1, 2]

        # Sample features at warped location. Features are [B, P, D] with
        # P = h*w; reshape to [B, D, h, w] for grid_sample.
        P = features.shape[1]
        h = int(round(P ** 0.5))
        if h * h != P:
            # Non-square patch grid → fall back to zero correction (stub).
            return torch.zeros(B, N, D, device=device, dtype=dtype)
        feat_chw = features.detach().permute(0, 2, 1).reshape(B, D, h, h)
        warped_feats = F.grid_sample(
            feat_chw, warped_grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        ).squeeze(-1).permute(0, 2, 1)              # [B, N, D]
        return warped_feats.detach()

    # --------------------------------------------------------------- forward
    def forward(
        self,
        slots: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        slot_centroids: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]]:
        """Hybrid forward: MAP assignment + ambiguity-gated correction.

        Returns:
            ``(corrected_final, existence_mask_final, aux)`` where
            ``aux = (z, merge_mask, count_logits)``. The triple matches the
            ``(slots, mask, aux)`` contract that
            :meth:`LatentProcessor.forward` expects from a Hungarian-style
            predictor; the ``aux`` tuple is forwarded to losses / logging via
            ``video.py`` (see spec §2 step 11 commentary).
        """
        # --- Step 1: MAP-with-reject ---
        map_out = self.map.forward(
            slots, prev_slots=prev_slots, existence_mask=existence_mask,
        )
        # The MAP module returns either a tensor, ``(reordered, weights)``, or
        # ``(reordered, mask, weights)`` depending on call signature. We always
        # pass ``existence_mask`` through, but be defensive in case a caller
        # invokes us without one.
        if isinstance(map_out, tuple):
            if len(map_out) == 3:
                reordered, existence_mask_updated, _ = map_out
            else:
                reordered, _ = map_out
                existence_mask_updated = self.map._last_existence_mask
        else:
            reordered = map_out
            existence_mask_updated = self.map._last_existence_mask

        B, N, D = reordered.shape
        device = reordered.device
        dtype = reordered.dtype

        # Default existence mask if none supplied: every slot alive. We still
        # return ``None`` at the end if the caller passed ``None``, so that the
        # public contract is unchanged.
        existence_mask_was_none = existence_mask_updated is None
        if existence_mask_was_none:
            existence_mask_updated = torch.ones(
                B, N, device=device, dtype=torch.bool,
            )
        else:
            existence_mask_updated = existence_mask_updated.to(device)

        # --- Step 2: extract per-slot LSAP margin ---
        margin_row = self.map.last_margin_per_slot  # [B, N]; detached
        # First frame: MAP did not solve LSAP. ``last_margin_per_slot`` is
        # ``None`` for legacy callers, or all +inf in this implementation. In
        # both cases there is nothing to gate, so skip every learnable branch
        # and return the MAP outputs unchanged. Aux tensors are zeros so the
        # caller's logging pipeline stays well-defined.
        is_first_frame = (
            margin_row is None
            or self.map.get_last_match_indices() is None
        )
        if is_first_frame:
            zero_aux = torch.zeros(B, N, device=device, dtype=dtype)
            count_logits = torch.zeros(B, N, device=device, dtype=dtype)
            if existence_mask_was_none:
                return reordered, None, (zero_aux, zero_aux, count_logits)
            return reordered, existence_mask_updated, (zero_aux, zero_aux, count_logits)

        margin_row = margin_row.to(device=device, dtype=dtype)
        # Replace +inf (padded / unmatched output positions) with a large
        # finite value so the sigmoid gate degenerates cleanly to ~0.
        large_finite = torch.tensor(1e6, device=device, dtype=dtype)
        margin_row = torch.where(
            torch.isfinite(margin_row), margin_row, large_finite
        )

        # --- Step 3: soft gate ---
        T_pos = self.T_gate_positive
        z_soft = torch.sigmoid((self.tau_gate - margin_row) / T_pos)  # [B, N]
        if self.training:
            z = z_soft
            route_mask_bool = z_soft > 0.0  # all positions active in training
        else:
            # Hard threshold at 0.5 honours T5' threshold policy and lets us
            # actually skip the evidence branches for confident slots.
            route_mask_bool = z_soft > 0.5
            z = route_mask_bool.to(dtype)

        any_routed = bool(route_mask_bool.any().item())

        # --- Step 4 & 5: cross-attention + motion ---
        attn_out = torch.zeros_like(reordered)
        wants_xattn = self.use_xattn and self.cross_attn is not None
        wants_motion = (
            self.use_motion
            and flow is not None
            and self.motion_projection is not None
        )

        if any_routed and (wants_xattn or wants_motion):
            # Build a query mask so cross-attention only contributes to routed
            # slots. ``key_padding_mask`` would mask key tokens; we instead
            # zero out the contribution post-hoc to honour the spec wording
            # (§2 step 4: "compute only if route_mask.any()").
            if wants_motion and (self.motion_standalone or wants_xattn):
                warped = self.warp_patches_by_flow(
                    flow=flow,
                    slots=reordered,
                    features=features,
                    slot_centroids=slot_centroids,
                )  # [B, N, D], detached
                motion_correction = self.motion_projection(warped)
            else:
                motion_correction = None

            if wants_motion and self.motion_standalone:
                # Ablation row: motion ONLY, replaces xattn entirely.
                attn_out = motion_correction * route_mask_bool.to(dtype).unsqueeze(-1)
            elif wants_xattn and features is not None:
                # Default path: cross-attend, optionally with motion-conditioned
                # query.
                if motion_correction is not None:
                    query_in = self.q_norm(reordered + motion_correction)
                else:
                    query_in = self.q_norm(reordered)
                key_in = self.kv_norm(features)
                attn_raw, _ = self.cross_attn(
                    query=query_in, key=key_in, value=key_in, need_weights=False,
                )
                # Residual + FFN block (pre-LN style around the FFN).
                hidden = self.attn_post_norm(reordered + attn_raw)
                attn_out = hidden + self.ffn(self.ffn_norm(hidden)) - reordered
                # Zero out attn_out at unrouted positions during inference so
                # the gated residual really collapses to identity for confident
                # slots.
                if not self.training:
                    attn_out = attn_out * route_mask_bool.to(dtype).unsqueeze(-1)
            elif wants_motion:
                # Motion-only fallback when cross-attention is disabled but
                # motion is enabled (ablation): treat the projection as the
                # correction directly.
                attn_out = motion_correction * route_mask_bool.to(dtype).unsqueeze(-1)

        # --- Step 6: gated residual ---
        corrected = reordered + z.unsqueeze(-1) * attn_out

        # --- Step 7: count head ---
        if self.use_count_head and self.count_mlp is not None:
            count_logits = self.count_mlp(corrected).squeeze(-1)  # [B, N]
        else:
            count_logits = torch.zeros(B, N, device=device, dtype=dtype)
        count_probs = torch.sigmoid(count_logits)

        # --- Step 8: cluster-merge with safety conditions ---
        merge_mask = torch.zeros(B, N, device=device, dtype=dtype)
        existence_mask_final = existence_mask_updated.clone()
        # ``cooldown`` is a *working copy* — we mutate it during the merge
        # decisions and persist the decremented version back at the end.
        cooldown: Dict[Tuple[int, int, int], int] = dict(self._cooldown_registry)

        if self.use_cluster_merge and N > 1:
            corrected_norm = F.normalize(corrected.detach(), dim=-1)
            for b in range(B):
                cos_b = corrected_norm[b] @ corrected_norm[b].T  # [N, N]
                # Only consider strict upper triangle pairs and only those
                # above the cosine threshold; sort by cosine descending so the
                # most-similar pairs win in the greedy walk.
                triu_idx = torch.triu_indices(N, N, offset=1, device=device)
                if triu_idx.numel() == 0:
                    continue
                cos_vals = cos_b[triu_idx[0], triu_idx[1]]
                keep = cos_vals > self.merge_cosine_threshold
                if not bool(keep.any().item()):
                    continue
                cand_i = triu_idx[0][keep].tolist()
                cand_j = triu_idx[1][keep].tolist()
                cand_c = cos_vals[keep].tolist()
                # Greedy descending walk over candidate pairs.
                order = sorted(
                    range(len(cand_c)), key=lambda k: cand_c[k], reverse=True
                )
                merged_local = set()  # local copy for O(1) lookup this batch
                for k in order:
                    i = int(cand_i[k])
                    j = int(cand_j[k])
                    if i in merged_local or j in merged_local:
                        continue
                    # Determine winner / loser by count probability (winner =
                    # higher count_probs). Spec uses (i, j) with j as winner;
                    # we honour that ordering by swapping when needed.
                    if count_probs[b, j] >= count_probs[b, i]:
                        loser, winner = i, j
                    else:
                        loser, winner = j, i

                    # Safety conditions (§5):
                    cond_winner_alive = bool(existence_mask_final[b, winner].item())
                    cond_winner_conf = (
                        count_probs[b, winner].item() > self.merge_count_winner_min
                    )
                    cond_loser_weak = (
                        count_probs[b, loser].item() < self.merge_count_loser_max
                    )
                    cond_winner_stronger = (
                        count_probs[b, winner].item() > count_probs[b, loser].item()
                    )
                    pair_id = self._pair_id(b, loser, winner)
                    cond_cooldown = cooldown.get(pair_id, 0) == 0

                    if (
                        cond_winner_alive
                        and cond_winner_conf
                        and cond_loser_weak
                        and cond_winner_stronger
                        and cond_cooldown
                    ):
                        # Pre-merge dormant snapshot, deduplicated by orig_idx.
                        self._dormant_snapshot_dedup(
                            b, loser, corrected[b, loser].detach().cpu()
                        )
                        merge_mask[b, loser] = 1.0
                        # Mask dtype may be bool, int, or float depending on
                        # caller; assign in the right dtype.
                        if existence_mask_final.dtype == torch.bool:
                            existence_mask_final[b, loser] = False
                        else:
                            existence_mask_final[b, loser] = 0
                        merged_local.add(loser)
                        merged_local.add(winner)
                        cooldown[pair_id] = self.merge_cooldown_frames

        # --- Step 8.5: decrement cooldowns and persist ---
        self._cooldown_registry = {
            k: v - 1 for k, v in cooldown.items() if v > 1
        }

        # --- Step 9: zero merged slots ---
        corrected_final = corrected * (1.0 - merge_mask).unsqueeze(-1)

        # --- Step 10: overwrite MAP state with the corrected outputs ---
        # Without this the next frame's LSAP would ignore our corrections and
        # any merge decisions (spec §4 HIGH #1 fix).
        self.map._prev_slots = corrected_final.detach()
        self.map._last_existence_mask = existence_mask_final

        # --- Step 11: birth-overflow hard failure ---
        if self.map._last_births_overflow:
            raise RuntimeError(
                "HybridPredictor: MAPWithReject reported birth overflow "
                "(more births than unclaimed slot positions). Increase N "
                "or check the data/config."
            )

        aux = (z, merge_mask, count_logits)
        if existence_mask_was_none:
            return corrected_final, None, aux
        return corrected_final, existence_mask_final, aux

    # ---------------------------------------------------------- merge helper
    def _dormant_snapshot_dedup(
        self, b: int, slot_idx: int, feature_cpu: torch.Tensor
    ) -> None:
        """Store a pre-merge snapshot in MAP's dormant registry.

        Deduplicated by ``orig_idx`` (only one dormant entry per original slot
        position) and stored on CPU to match MAP's existing convention
        (spec §5, condition 6 + 7).
        """
        if self.map._dormant is None:
            return
        registry = self.map._dormant[b]
        # Drop any pre-existing dormant entry for the same orig_idx so the
        # registry stays single-valued per slot position.
        registry[:] = [d for d in registry if d.get("orig_idx", -1) != slot_idx]
        registry.append({
            "feature": feature_cpu,
            "age": 0,
            "orig_idx": int(slot_idx),
        })


class NullAwareMemoryPredictor(nn.Module):
    """Open-set slot routing predictor (Track B, TRACK_B_SPEC v10).

    Each query slot is routed over {active_prev_slots, dormant_tokens, null}.
    Patches are EVIDENCE only: they condition the query via cross-attention and
    are NEVER part of the candidate pool. Birth is a pure NULL-branch
    consequence: null wins AND a residual-evidence peak exceeds
    ``residual_birth_threshold`` for that slot's target patch.

    Differentiability:
      * ``prev_slots`` and dormant features are detached (identity memory).
      * The learned null token is a single ``nn.Parameter``.
      * Pointer-transformer + patch-conditioning weights learn via gradient
        flow through ``updated`` into the downstream reconstruction losses.
      * Birth-spawned slots are detached from the patch pool so no gradient
        flows back into the feature extractor.

    Drop-in for :class:`LatentProcessor`: ``_hungarian_match = None`` routes us
    through the Hungarian-style dispatch path in ``video.py``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ffn_dim: int = 512,
        attn_dropout: float = 0.0,
        max_dormant: int = 32,
        max_dormant_age: int = 10,
        age_penalty: float = 0.05,
        residual_birth_threshold: float = 0.7,
        residual_birth_window: int = 3,
        temperature: float = 1.0,
        pre_match: bool = False,
        # Ablation switches
        use_dormant: bool = True,
        use_null: bool = True,
        use_residual_birth: bool = True,
        use_patch_conditioning: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.max_dormant = int(max_dormant)
        self.max_dormant_age = int(max_dormant_age)
        self.age_penalty = float(age_penalty)
        self.residual_birth_threshold = float(residual_birth_threshold)
        self.residual_birth_window = int(residual_birth_window)
        self.temperature = float(temperature)
        self.pre_match = bool(pre_match)
        self.use_dormant = bool(use_dormant)
        self.use_null = bool(use_null)
        self.use_residual_birth = bool(use_residual_birth)
        self.use_patch_conditioning = bool(use_patch_conditioning)

        # Sentinel: LatentProcessor dispatches Hungarian-style when this attr exists.
        self._hungarian_match = None

        # --- Learned null token (single D-dim parameter, broadcast per batch) ---
        self.null_token = nn.Parameter(torch.zeros(self.dim))

        # --- Patch-evidence conditioning block (Step 3a) ---
        if self.use_patch_conditioning:
            self.patch_q_norm = nn.LayerNorm(self.dim)
            self.patch_kv_norm = nn.LayerNorm(self.dim)
            self.patch_cross_attn = nn.MultiheadAttention(
                embed_dim=self.dim,
                num_heads=self.num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            self.query_cond_norm = nn.LayerNorm(self.dim)
        else:
            self.patch_q_norm = None
            self.patch_kv_norm = None
            self.patch_cross_attn = None
            self.query_cond_norm = None

        # --- Pointer transformer (Step 3b): slot queries attend over
        #     concatenated identity candidates (active + dormant + null). ---
        self.pointer_q_norm = nn.LayerNorm(self.dim)
        self.pointer_kv_norm = nn.LayerNorm(self.dim)
        self.pointer_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.pointer_ffn = nn.Sequential(
            nn.Linear(self.dim, int(ffn_dim)),
            nn.GELU(),
            nn.Linear(int(ffn_dim), self.dim),
        )
        self.pointer_ffn_norm = nn.LayerNorm(self.dim)

        # --- State (per-video; cleared by ``reset``) ---
        self._prev_slots: Optional[torch.Tensor] = None           # [B, N, D] detached
        self._dormant: Optional[List[List[Dict[str, Any]]]] = None
        self._last_match_indices: Optional[torch.Tensor] = None
        self._last_existence_mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ utils
    def reset(self) -> None:
        """Reset per-video state."""
        self._prev_slots = None
        self._dormant = None
        self._last_match_indices = None
        self._last_existence_mask = None

    def get_last_match_indices(self) -> Optional[torch.Tensor]:
        return self._last_match_indices

    def match_to_reference(self, slots: torch.Tensor) -> torch.Tensor:
        """V1 interface for pre-matching mode."""
        out = self.forward(slots)
        return out if isinstance(out, torch.Tensor) else out[0]

    # --------------------------------------------------------- dormant helpers
    def _select_topM_dormant(
        self, b: int, query_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Retrieve up to ``max_dormant`` dormant tokens for batch ``b``.

        Returns ``(feats, ages, orig_indices_used)`` where ``feats`` has shape
        ``[M_used, D]`` (possibly 0). Selection ranks entries by
        ``max_over_queries(cos_sim) - age_penalty * age`` and takes the top
        ``max_dormant``. No gradient flows through dormant features.
        """
        device = query_feats.device
        dtype = query_feats.dtype
        if not self.use_dormant or self._dormant is None or len(self._dormant[b]) == 0:
            return (
                torch.zeros(0, self.dim, device=device, dtype=dtype),
                torch.zeros(0, device=device, dtype=dtype),
                [],
            )
        entries = self._dormant[b]
        feats = torch.stack([e["feature"] for e in entries]).to(
            device=device, dtype=dtype
        )  # [K, D]
        ages = torch.tensor(
            [e["age"] for e in entries], device=device, dtype=dtype
        )  # [K]
        if feats.shape[0] <= self.max_dormant:
            return feats.detach(), ages.detach(), list(range(feats.shape[0]))
        # Over cap: rank by age-penalized best-query similarity, keep top-M.
        q_norm = F.normalize(query_feats, dim=-1)      # [N, D]
        f_norm = F.normalize(feats, dim=-1)            # [K, D]
        sim = (q_norm @ f_norm.T).max(dim=0).values    # [K]
        score = sim - self.age_penalty * ages
        topk = torch.topk(score, k=self.max_dormant, largest=True).indices.tolist()
        topk_sorted = sorted(topk)
        sel_feats = feats[topk_sorted].detach()
        sel_ages = ages[topk_sorted].detach()
        return sel_feats, sel_ages, topk_sorted

    def _compute_residual_scores(
        self,
        slots_aligned: torch.Tensor,
        features: torch.Tensor,
        existence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Residual-evidence score per patch (higher = more unexplained).

        Simplest fallback implementation per spec §7: since we do not have the
        corrector's slot-to-patch attention inside the predictor, approximate
        coverage as the best cosine similarity from any alive slot to that
        patch. ``residual[b, p] = 1 - max_k cos(slot[b, k], patch[b, p])``.
        """
        # slots_aligned: [B, N, D]; features: [B, P, D]; existence_mask: [B, N]
        s_norm = F.normalize(slots_aligned, dim=-1)
        f_norm = F.normalize(features, dim=-1)
        sim = torch.einsum("bnd,bpd->bnp", s_norm, f_norm)  # [B, N, P]
        alive = existence_mask.to(dtype=sim.dtype).unsqueeze(-1)  # [B, N, 1]
        # Dead slots contribute no coverage.
        sim = sim * alive + (alive - 1.0) * 1e4
        coverage = sim.max(dim=1).values                    # [B, P]
        return 1.0 - coverage                               # [B, P]

    def _spawn_from_residual_peak(
        self,
        features_b: torch.Tensor,
        peak_pos: int,
    ) -> torch.Tensor:
        """Mean-pool patches in a window around ``peak_pos`` and L2-normalize.

        Operates on a detached copy of the patch pool so gradient does not flow
        back into the feature extractor.
        """
        P, D = features_b.shape
        h = int(round(P ** 0.5))
        feats = features_b.detach()
        if h * h == P:
            # Square grid: pool a (window x window) neighborhood.
            w = self.residual_birth_window
            half = w // 2
            y, x = divmod(int(peak_pos), h)
            ys = torch.arange(max(0, y - half), min(h, y + half + 1), device=feats.device)
            xs = torch.arange(max(0, x - half), min(h, x + half + 1), device=feats.device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            idx = (yy * h + xx).reshape(-1)
            pool = feats.index_select(0, idx).mean(dim=0)
        else:
            # Non-square grid: fall back to a linear window around ``peak_pos``.
            w = self.residual_birth_window
            half = w // 2
            lo = max(0, int(peak_pos) - half)
            hi = min(P, int(peak_pos) + half + 1)
            pool = feats[lo:hi].mean(dim=0)
        return F.normalize(pool, dim=-1, eps=1e-8)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        slots: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]]:
        """Open-set routed forward.

        Args:
            slots: ``[B, N, D]`` corrector output at current frame t.
            prev_slots: optional override for internal ``self._prev_slots``.
            existence_mask: ``[B, N]`` alive flag (bool/float).
            features: ``[B, P, D]`` current-frame patch features (required
                for patch conditioning and residual birth; if ``None`` those
                paths degrade to no-ops).
        Returns:
            ``(slots_aligned, existence_mask_updated,
                (route_logits, birth_mask, dormant_activity))``.
        """
        B, N, D = slots.shape
        device = slots.device
        dtype = slots.dtype

        # Normalize incoming existence mask.
        existence_mask_was_none = existence_mask is None
        if existence_mask_was_none:
            existence_mask_in = torch.ones(B, N, device=device, dtype=torch.bool)
        else:
            existence_mask_in = existence_mask.to(device)

        # --- Step 1: First-frame bootstrap ---
        reference = prev_slots if prev_slots is not None else self._prev_slots
        if reference is None:
            self._prev_slots = slots.detach()
            self._dormant = [[] for _ in range(B)]
            self._last_match_indices = None
            self._last_existence_mask = existence_mask_in
            zero_logits = torch.zeros(
                B, N, N + self.max_dormant + 1, device=device, dtype=dtype
            )
            zero_birth = torch.zeros(B, N, device=device, dtype=dtype)
            zero_activity = torch.zeros(B, self.max_dormant, device=device, dtype=dtype)
            aux = (zero_logits, zero_birth, zero_activity)
            if existence_mask_was_none:
                return slots, None, aux
            return slots, existence_mask_in, aux

        # --- Step 2: Build candidate pool per batch ---
        M_cap = self.max_dormant
        active_tokens = reference.detach().to(device=device, dtype=dtype)  # [B, N, D]
        dormant_feats_list: List[torch.Tensor] = []
        dormant_ages_list: List[torch.Tensor] = []
        dormant_mask_list: List[torch.Tensor] = []
        dormant_orig_used: List[List[int]] = []
        for b in range(B):
            feats_b, ages_b, used_b = self._select_topM_dormant(
                b, active_tokens[b]
            )
            k = feats_b.shape[0]
            pad_k = M_cap - k
            if pad_k > 0:
                feats_b = torch.cat(
                    [feats_b, torch.zeros(pad_k, D, device=device, dtype=dtype)], dim=0
                )
                ages_b = torch.cat(
                    [ages_b, torch.zeros(pad_k, device=device, dtype=dtype)], dim=0
                )
            mask_b = torch.zeros(M_cap, device=device, dtype=torch.bool)
            if k > 0:
                mask_b[:k] = True
            dormant_feats_list.append(feats_b)
            dormant_ages_list.append(ages_b)
            dormant_mask_list.append(mask_b)
            dormant_orig_used.append(used_b)
        dormant_tokens = torch.stack(dormant_feats_list, dim=0)   # [B, M, D]
        dormant_ages = torch.stack(dormant_ages_list, dim=0)      # [B, M]
        dormant_valid = torch.stack(dormant_mask_list, dim=0)     # [B, M]

        null_token_b = self.null_token.to(dtype=dtype).view(1, 1, D).expand(B, 1, D)
        cand_tokens = torch.cat(
            [active_tokens, dormant_tokens, null_token_b], dim=1
        )  # [B, N+M+1, D]

        # --- Step 3a: Patch-evidence conditioning (optional) ---
        if self.use_patch_conditioning and features is not None and self.patch_cross_attn is not None:
            q_in = self.patch_q_norm(slots)
            kv_in = self.patch_kv_norm(features)
            patch_cond, _ = self.patch_cross_attn(
                query=q_in, key=kv_in, value=kv_in, need_weights=False
            )
            query_cond = self.query_cond_norm(slots + patch_cond)
        else:
            query_cond = slots

        # --- Step 3b: Pointer scoring over candidates ---
        q_ptr = self.pointer_q_norm(query_cond)
        kv_ptr = self.pointer_kv_norm(cand_tokens)
        # Mask out padded dormant slots and (optionally) the null column.
        key_padding_mask = torch.zeros(
            B, N + M_cap + 1, device=device, dtype=torch.bool
        )
        # Padded dormant positions -> mask out.
        key_padding_mask[:, N:N + M_cap] = ~dormant_valid
        if not self.use_null:
            key_padding_mask[:, N + M_cap] = True  # ablation: remove null
        attn_out, attn_weights = self.pointer_attn(
            query=q_ptr,
            key=kv_ptr,
            value=kv_ptr,
            need_weights=True,
            average_attn_weights=True,
            key_padding_mask=key_padding_mask,
        )
        # `attn_weights` returned by nn.MultiheadAttention are SOFTMAXed head-
        # averaged probabilities over the candidate axis. The spec reads them
        # as route scores, but applying another /temperature softmax on top
        # smears training-time probabilities toward uniform over N+M+1 and
        # produces a nearly-identity `updated = einsum(route_probs, cand)`
        # — Codex round-24 L1. Fix: convert attn_weights to log-logits first
        # so the downstream softmax operates on a proper logit tensor.
        _ = self.pointer_ffn_norm(attn_out + self.pointer_ffn(attn_out))
        route_logits = torch.log(attn_weights.clamp_min(1e-10))  # [B, N, N+M+1]

        # Age penalty on dormant columns only (Step 3b cont.).
        if self.use_dormant and M_cap > 0:
            penalty = self.age_penalty * dormant_ages.unsqueeze(1)  # [B, 1, M]
            route_logits = route_logits.clone()
            route_logits[:, :, N:N + M_cap] = (
                route_logits[:, :, N:N + M_cap] - penalty
            )

        # Enforce masks on dormant and null ablation columns before softmax.
        neg_inf = torch.finfo(route_logits.dtype).min
        inval_cols = key_padding_mask.unsqueeze(1).expand(-1, N, -1)  # [B, N, K]
        route_logits = route_logits.masked_fill(inval_cols, neg_inf)
        # Also forbid routing to dormant columns entirely when dormant is off.
        if not self.use_dormant and M_cap > 0:
            route_logits[:, :, N:N + M_cap] = neg_inf

        # --- Step 4: Softmax → route probabilities ---
        route_probs = F.softmax(route_logits / self.temperature, dim=-1)  # [B, N, K]

        # --- Step 5: Per-slot winner categorization ---
        argmax_route = route_probs.argmax(dim=-1)                  # [B, N]
        null_col = N + M_cap
        null_routed_mask = (argmax_route == null_col)              # [B, N] bool
        cont_mask = argmax_route < N                               # CONTINUE
        reenter_mask = (argmax_route >= N) & (argmax_route < null_col)

        # --- Step 6: Soft update via expected candidate ---
        if self.training:
            updated = torch.einsum("bnk,bkd->bnd", route_probs, cand_tokens)
        else:
            # Hard-argmax path at inference.
            idx = argmax_route.unsqueeze(-1).expand(-1, -1, D)      # [B, N, D]
            updated = cand_tokens.gather(1, idx)                    # [B, N, D]
        # Zero out null-routed positions; birth may fill them below.
        updated = updated * (~null_routed_mask).to(dtype=updated.dtype).unsqueeze(-1)

        # --- Step 7: Residual-peak birth (NULL-BRANCH CONSEQUENCE ONLY) ---
        birth_mask = torch.zeros(B, N, device=device, dtype=dtype)
        if (
            self.use_residual_birth
            and self.use_null
            and features is not None
            and bool(null_routed_mask.any().item())
        ):
            # Alive proxy for residual coverage: use existence_mask_in AND
            # (not null-routed) so null positions don't self-explain coverage.
            alive_for_resid = existence_mask_in.to(dtype=torch.bool) & (~null_routed_mask)
            residual = self._compute_residual_scores(
                updated, features, alive_for_resid.to(dtype=dtype)
            )  # [B, P]
            for b in range(B):
                nulls_b = torch.where(null_routed_mask[b])[0]
                if nulls_b.numel() == 0:
                    continue
                # Top-(len(nulls_b)) peaks that also exceed threshold.
                k = int(nulls_b.numel())
                topk = torch.topk(residual[b], k=min(k, residual.shape[1]), largest=True)
                peak_positions = topk.indices.tolist()
                peak_values = topk.values.tolist()
                claimed_positions: set = set()
                for slot_idx, pos, val in zip(
                    nulls_b.tolist(), peak_positions, peak_values
                ):
                    if val <= self.residual_birth_threshold:
                        continue
                    if pos in claimed_positions:
                        continue
                    spawned = self._spawn_from_residual_peak(features[b], pos)
                    updated[b, slot_idx] = spawned.to(dtype=updated.dtype)
                    birth_mask[b, slot_idx] = 1.0
                    claimed_positions.add(pos)

        # --- Step 8: Existence-mask update ---
        existence_mask_final = torch.zeros(B, N, device=device, dtype=torch.bool)
        # CONTINUE: inherit the alive flag of the matched active position.
        if cont_mask.any():
            # argmax_route[b, n] is the prev-slot index; its alive flag is
            # existence_mask_in[b, argmax_route[b, n]].
            cont_src = argmax_route.clamp(max=N - 1)
            cont_alive = existence_mask_in.gather(1, cont_src)  # [B, N]
            existence_mask_final = existence_mask_final | (cont_mask & cont_alive.to(torch.bool))
        # RE-ENTER: mark alive unconditionally (dormant has no existence flag).
        existence_mask_final = existence_mask_final | reenter_mask
        # NULL + birth: alive; NULL + no birth: dead.
        existence_mask_final = existence_mask_final | (birth_mask > 0)

        # --- Step 9: Dormant registry update ---
        # Active positions that were NOT continued → push to dormant.
        # Dormant entries that were matched (re-entered) → pop from dormant.
        # Age remaining entries; evict > max_dormant_age.
        dormant_activity = torch.zeros(B, self.max_dormant, device=device, dtype=dtype)
        if self._dormant is None:
            self._dormant = [[] for _ in range(B)]
        for b in range(B):
            # Which active orig-idx columns were used as CONTINUE?
            cont_cols_b = argmax_route[b][cont_mask[b]].tolist()
            matched_active = set(int(c) for c in cont_cols_b)
            # Which dormant local indices (within the top-M pool) were used?
            reenter_cols_b = argmax_route[b][reenter_mask[b]].tolist()
            matched_local_dormant = set(
                int(c) - N for c in reenter_cols_b if 0 <= int(c) - N < M_cap
            )
            # Log activity: count of dormant hits this step (per local index).
            for lidx in matched_local_dormant:
                if 0 <= lidx < self.max_dormant:
                    dormant_activity[b, lidx] = 1.0

            used_map = dormant_orig_used[b]                 # local_idx -> registry_idx
            matched_registry_idx = {
                used_map[lidx] for lidx in matched_local_dormant
                if 0 <= lidx < len(used_map)
            }

            # Pop matched dormant entries; age and evict the rest.
            old_registry = self._dormant[b]
            new_registry: List[Dict[str, Any]] = []
            for reg_idx, entry in enumerate(old_registry):
                if reg_idx in matched_registry_idx:
                    continue  # re-activated
                entry = dict(entry)
                entry["age"] = int(entry.get("age", 0)) + 1
                if entry["age"] <= self.max_dormant_age:
                    new_registry.append(entry)

            # Push unmatched active identities (those from prev that no current
            # slot CONTINUEs to) into the dormant registry. Deduplicate by
            # orig_idx in case the same active position was already dormant.
            for j in range(N):
                if j in matched_active:
                    continue
                if not bool(existence_mask_in[b, j].item()):
                    continue
                feat_cpu = active_tokens[b, j].detach().cpu()
                new_registry[:] = [
                    d for d in new_registry if d.get("orig_idx", -1) != int(j)
                ]
                new_registry.append({
                    "feature": feat_cpu,
                    "age": 0,
                    "orig_idx": int(j),
                })

            # Cap registry size defensively.
            if len(new_registry) > self.max_dormant * 2:
                new_registry.sort(key=lambda d: int(d.get("age", 0)))
                new_registry = new_registry[: self.max_dormant * 2]
            self._dormant[b] = new_registry

        # --- Step 10: State persistence ---
        self._prev_slots = updated.detach()
        # last_match_indices: CONTINUE → source row; else -1 sentinel.
        last_match = torch.full((B, N), -1, device=device, dtype=torch.long)
        last_match[cont_mask] = argmax_route[cont_mask].to(torch.long)
        self._last_match_indices = last_match
        self._last_existence_mask = existence_mask_final

        aux = (route_logits.detach(), birth_mask, dormant_activity)
        if existence_mask_was_none:
            return updated, None, aux
        return updated, existence_mask_final, aux


class DepthAugmentedMAPPredictor(nn.Module):
    """MAP-with-reject predictor with depth-histogram-augmented LSAP cost (idea #024).

    Wraps the same MAP-with-reject algorithm as :class:`MAPWithRejectPredictor`
    but augments the per-pair assignment cost with a *depth-histogram* distance
    so that two slots can only match if their support regions live at similar
    depths across frames. The motivation is temporal consistency: visually
    similar slots that sit at very different depths (e.g. an object vs. its
    shadow on the ground) should NOT be merged into the same track.

    Depth is treated as a **frozen input**:
      * No depth model is trained — `depth` tensors come from Kubric GT
        (MOVi-D shards) or MegaSAM (YT-VIS re-sharded shards), see
        :mod:`data.save_ytvis2021`.
      * Histograms are built on a detached copy of ``depth`` so no gradient
        flows into the depth source.
      * When ``lambda_depth == 0`` the forward is numerically identical to
        :class:`MAPWithRejectPredictor` (explicit disable — the depth-free
        branch is only taken in this case).
      * When ``lambda_depth > 0`` we honor the project's no-fallback contract:
        missing ``depth`` (or missing ``slot_masks``, which is needed to build
        per-slot histograms) raises :class:`RuntimeError` so a misconfigured
        pipeline fails loudly rather than silently training depth-free.

    The ``requires_depth = True`` class attribute is the advertised depth-
    contract flag consumed by upstream dispatchers (mirroring the same flag on
    :class:`DepthEdgeFeatureInit`).

    Interface contract matches :class:`MAPWithRejectPredictor` and
    :class:`NullAwareMemoryPredictor`: the sentinel ``_hungarian_match = None``
    attribute tells :class:`LatentProcessor` to route us through the
    Hungarian-style dispatch path in ``video.py``.
    """

    # Advertise depth dependency so upstream wiring (dispatcher / config-time
    # checks) can enforce that `inputs["depth"]` is threaded through. Combined
    # with the hard-raise inside `forward`, this forbids silent fallback.
    requires_depth = True

    def __init__(
        self,
        dim: int,
        similarity: str = "cosine",
        reject_threshold: float = 0.5,
        age_penalty: float = 0.05,
        max_dormant_age: int = 10,
        lambda_appearance: float = 1.0,
        lambda_depth: float = 0.2,
        depth_hist_bins: int = 10,
        pre_match: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.similarity = similarity
        self.reject_threshold = float(reject_threshold)
        self.age_penalty = float(age_penalty)
        self.max_dormant_age = int(max_dormant_age)
        self.lambda_appearance = float(lambda_appearance)
        self.lambda_depth = float(lambda_depth)
        self.depth_hist_bins = int(depth_hist_bins)
        self.pre_match = bool(pre_match)

        if self.depth_hist_bins < 1:
            raise ValueError(
                f"depth_hist_bins must be >= 1, got {self.depth_hist_bins}"
            )

        # Sentinel so LatentProcessor dispatches Hungarian-style.
        self._hungarian_match = None

        # --- Per-video state (cleared by ``reset``) ---
        self._prev_slots: Optional[torch.Tensor] = None           # [B, N, D]
        self._dormant: Optional[List[List[Dict]]] = None           # per-batch dormant list
        self._last_match_indices: Optional[torch.Tensor] = None
        self._next_id: Optional[List[int]] = None
        self._last_margin_per_slot: Optional[torch.Tensor] = None  # [B, N]
        self._last_births_overflow: bool = False
        self._last_existence_mask: Optional[torch.Tensor] = None
        # Per-slot depth histogram from the previous frame: [B, N, bins] or None.
        self._prev_hist_by_slot: Optional[torch.Tensor] = None

    @property
    def last_margin_per_slot(self) -> Optional[torch.Tensor]:
        return self._last_margin_per_slot

    def reset(self) -> None:
        """Reset per-video state."""
        self._prev_slots = None
        self._dormant = None
        self._last_match_indices = None
        self._next_id = None
        self._last_margin_per_slot = None
        self._last_births_overflow = False
        self._last_existence_mask = None
        self._prev_hist_by_slot = None

    def get_last_match_indices(self) -> Optional[torch.Tensor]:
        return self._last_match_indices

    def match_to_reference(self, slots: torch.Tensor) -> torch.Tensor:
        """V1 interface for pre-matching mode."""
        out = self.forward(slots)
        return out if isinstance(out, torch.Tensor) else out[0]

    # -------------------------------------------------------- depth helpers
    @staticmethod
    def _resize_masks_to_depth(
        slot_masks: torch.Tensor, depth_hw: Tuple[int, int]
    ) -> torch.Tensor:
        """Bilinearly resize ``[B, N, Hm, Wm]`` masks to depth resolution.

        No-op when the spatial dimensions already match. Detached — the depth
        cost term does not back-propagate into the corrector's mask logits.
        """
        B, N, Hm, Wm = slot_masks.shape
        Hd, Wd = depth_hw
        if Hm == Hd and Wm == Wd:
            return slot_masks.detach()
        flat = slot_masks.detach().reshape(B * N, 1, Hm, Wm).to(torch.float32)
        resized = F.interpolate(
            flat, size=(Hd, Wd), mode="bilinear", align_corners=False
        )
        return resized.reshape(B, N, Hd, Wd)

    def _compute_slot_histograms(
        self,
        depth: torch.Tensor,
        slot_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mask-weighted depth histograms per slot.

        Args:
            depth: ``[B, H, W]`` current-frame depth (detached input).
            slot_masks: ``[B, N, H, W]`` non-negative soft masks.

        Returns:
            ``[B, N, depth_hist_bins]`` L1-normalized histograms. Slots with
            zero mass get a uniform histogram so chi-square / L1 distances
            against other slots stay bounded (but are dominated by histograms
            with real mass).
        """
        depth_d = depth.detach()
        masks_d = slot_masks.detach()
        B, H, W = depth_d.shape
        N = masks_d.shape[1]
        device = depth_d.device
        dtype = masks_d.dtype

        # Per-frame (per-batch) min/max normalization → [0, 1] bin coordinate.
        flat_depth = depth_d.reshape(B, -1).to(torch.float32)         # [B, HW]
        dmin = flat_depth.min(dim=1, keepdim=True).values             # [B, 1]
        dmax = flat_depth.max(dim=1, keepdim=True).values             # [B, 1]
        denom = (dmax - dmin).clamp_min(1e-6)
        norm = (flat_depth - dmin) / denom                             # [B, HW]
        # Bin index in [0, bins-1].
        bins = self.depth_hist_bins
        bin_idx = (norm * bins).clamp(max=bins - 1).floor().long()    # [B, HW]

        # Build one-hot representation [B, HW, bins] once; reuse across slots.
        one_hot = F.one_hot(bin_idx, num_classes=bins).to(dtype)      # [B, HW, bins]

        masks_flat = masks_d.reshape(B, N, H * W).clamp_min(0.0).to(dtype)  # [B, N, HW]
        # hist[b, k, c] = sum_p masks[b, k, p] * one_hot[b, p, c]
        hist = torch.einsum("bnp,bpc->bnc", masks_flat, one_hot)       # [B, N, bins]
        # L1-normalize. Empty slots get a uniform distribution so distances
        # remain finite; they are typically gated out by existence_mask.
        totals = hist.sum(dim=-1, keepdim=True)                         # [B, N, 1]
        uniform = torch.full_like(hist, 1.0 / float(bins))
        hist = torch.where(totals > 1e-8, hist / totals.clamp_min(1e-8), uniform)
        return hist

    @staticmethod
    def _hist_l1_distance(
        hist_cur: torch.Tensor, hist_prev: torch.Tensor
    ) -> torch.Tensor:
        """Pairwise L1 distance between current-frame and reference histograms.

        Args:
            hist_cur: ``[N, bins]``.
            hist_prev: ``[K, bins]``.

        Returns:
            ``[N, K]`` pairwise L1 distance in ``[0, 2]`` (since both are
            probability vectors). Divided by 2 so the term lives in ``[0, 1]``
            — compatible scale with cosine distance.
        """
        diff = hist_cur.unsqueeze(1) - hist_prev.unsqueeze(0)  # [N, K, bins]
        return 0.5 * diff.abs().sum(dim=-1)                    # [N, K]

    # --------------------------------------------------------------- forward
    def forward(
        self,
        slots: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        slot_masks: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        **kwargs,
    ):
        """Depth-augmented MAP-with-reject forward.

        Args:
            slots: ``[B, N, D]`` current-frame slot features.
            prev_slots: optional override for ``self._prev_slots``.
            existence_mask: ``[B, N]`` alive flag (bool/float).
            features: unused; accepted for interface parity with
                :class:`NullAwareMemoryPredictor` / :class:`HybridPredictor`.
            depth: ``[B, H, W]`` current-frame depth map (frozen, detached).
                When ``None`` the depth term is skipped.
            slot_masks: ``[B, N, Hm, Wm]`` slot-to-patch soft masks (from the
                corrector). Resized bilinearly to match ``depth`` spatially.
                When ``None`` the depth term is skipped.

        Returns:
            Same contract as :class:`MAPWithRejectPredictor`: either
            ``reordered`` (tensor), ``(reordered, mask, None)`` if
            ``existence_mask`` was provided, or ``(reordered, None)`` if
            ``return_weights=True`` and no mask was passed.
        """
        from scipy.optimize import linear_sum_assignment

        B, N, D = slots.shape
        device = slots.device
        reference = prev_slots if prev_slots is not None else self._prev_slots

        # --- Depth term availability ---
        # No silent fallback: when the user asked for a depth-augmented cost
        # (``lambda_depth > 0``) we require the full depth payload. A stale
        # cache, a missing ``depth`` key, or a predictor configured ahead of
        # its data pipeline would otherwise train a *different* model with
        # ``use_depth=False`` and no failure signal. Only ``lambda_depth == 0``
        # (explicit disable) takes the depth-free branch.
        if self.lambda_depth != 0.0:
            if depth is None:
                raise RuntimeError(
                    f"{type(self).__name__} was configured with "
                    f"lambda_depth={self.lambda_depth} (>0) but received "
                    f"depth=None. No fallback is allowed: thread `depth` from "
                    f"`inputs['depth']` through the processor, or set "
                    f"lambda_depth=0 to explicitly disable the depth term."
                )
            if slot_masks is None:
                raise RuntimeError(
                    f"{type(self).__name__} was configured with "
                    f"lambda_depth={self.lambda_depth} (>0) but received "
                    f"slot_masks=None. The histogram term requires per-slot "
                    f"soft masks from the corrector; pass `slot_masks=curr_masks` "
                    f"from the processor call site, or set lambda_depth=0."
                )
            use_depth = True
        else:
            use_depth = False

        # Precompute current-frame histograms once per call.
        cur_hist: Optional[torch.Tensor] = None
        if use_depth:
            # Align masks to depth resolution, then compute histograms.
            dH, dW = int(depth.shape[-2]), int(depth.shape[-1])
            masks_aligned = self._resize_masks_to_depth(slot_masks, (dH, dW))
            cur_hist = self._compute_slot_histograms(depth, masks_aligned)  # [B, N, bins]

        # --- First frame: initialize state ---
        if reference is None:
            self._prev_slots = slots.detach()
            self._dormant = [[] for _ in range(B)]
            self._next_id = [N for _ in range(B)]
            self._last_match_indices = None
            self._last_margin_per_slot = torch.full(
                (B, N), float("inf"), device=device, dtype=slots.dtype
            )
            self._last_births_overflow = False
            self._last_existence_mask = existence_mask
            # Stash first-frame histograms so frame 2's cost sees them.
            self._prev_hist_by_slot = (
                cur_hist.detach() if cur_hist is not None else None
            )
            if existence_mask is not None:
                return slots, existence_mask, None
            return slots if not return_weights else (slots, None)

        # --- Build per-batch augmented cost matrix and solve LSAP ---
        ref_norm = F.normalize(reference, dim=-1)
        cur_norm = F.normalize(slots, dim=-1)

        reordered_list: List[torch.Tensor] = []
        indices_list: List[torch.Tensor] = []
        margin_list: List[torch.Tensor] = []
        new_hist_by_slot: Optional[List[torch.Tensor]] = (
            [] if cur_hist is not None else None
        )
        self._last_births_overflow = False

        prev_hist = self._prev_hist_by_slot  # [B_prev, N, bins] or None

        for b in range(B):
            active_feats = ref_norm[b]                 # [N, D]
            dormant_entries = self._dormant[b]
            n_active = N
            n_dormant = len(dormant_entries)
            n_cols = n_active + n_dormant
            cur_feats = cur_norm[b]                     # [N, D]

            # --- Appearance cost (cosine distance) ---
            active_cost = 1.0 - (cur_feats @ active_feats.T)  # [N, N]

            if existence_mask is not None:
                cur_invalid = ~existence_mask[b].bool()
                prev_invalid = cur_invalid  # fixed K
                active_cost[cur_invalid, :] = 1e6
                active_cost[:, prev_invalid] = 1e6

            if n_dormant > 0:
                dormant_feats = torch.stack(
                    [d["feature"] for d in dormant_entries]
                ).to(device)
                dormant_feats = F.normalize(dormant_feats, dim=-1)
                dormant_cost = 1.0 - (cur_feats @ dormant_feats.T)  # [N, n_dormant]
                ages = torch.tensor(
                    [d["age"] for d in dormant_entries],
                    device=device, dtype=torch.float32,
                )
                dormant_cost = dormant_cost + self.age_penalty * ages.unsqueeze(0)
            else:
                dormant_cost = None

            # --- Depth-histogram term ---
            depth_active = None
            depth_dormant = None
            if (
                use_depth
                and cur_hist is not None
                and prev_hist is not None
                and b < prev_hist.shape[0]
            ):
                h_cur_b = cur_hist[b].to(device)                          # [N, bins]
                h_prev_b = prev_hist[b].to(device)                        # [N, bins]
                depth_active = self._hist_l1_distance(h_cur_b, h_prev_b)  # [N, N]

                if n_dormant > 0:
                    # Dormant entries carry a cached histogram when available.
                    dhist_list = []
                    for d_entry in dormant_entries:
                        dh = d_entry.get("hist", None)
                        if dh is None:
                            dhist_list.append(
                                torch.full(
                                    (self.depth_hist_bins,),
                                    1.0 / float(self.depth_hist_bins),
                                    device=device, dtype=h_cur_b.dtype,
                                )
                            )
                        else:
                            dhist_list.append(dh.to(device=device, dtype=h_cur_b.dtype))
                    dhist = torch.stack(dhist_list, dim=0)                 # [n_dormant, bins]
                    depth_dormant = self._hist_l1_distance(h_cur_b, dhist) # [N, n_dormant]

            # --- Compose weighted cost ---
            # For masked (1e6) entries we keep them unchanged — adding a bounded
            # depth term to 1e6 still reads as "invalid".
            if depth_active is not None:
                active_cost = (
                    self.lambda_appearance * active_cost
                    + self.lambda_depth * depth_active
                )
            else:
                active_cost = self.lambda_appearance * active_cost

            if dormant_cost is not None:
                if depth_dormant is not None:
                    dormant_cost = (
                        self.lambda_appearance * dormant_cost
                        + self.lambda_depth * depth_dormant
                    )
                else:
                    dormant_cost = self.lambda_appearance * dormant_cost
                cost = torch.cat([active_cost, dormant_cost], dim=1)
            else:
                cost = active_cost

            # --- Pad to square, solve LSAP ---
            n_rows = N
            if n_cols > n_rows:
                pad = torch.full((n_cols - n_rows, n_cols), 1e6, device=device)
                cost_sq = torch.cat([cost, pad], dim=0)
            elif n_rows > n_cols:
                pad = torch.full(
                    (n_rows, n_rows - n_cols),
                    self.reject_threshold + 0.01,
                    device=device,
                )
                cost_sq = torch.cat([cost, pad], dim=1)
            else:
                cost_sq = cost

            cost_np = cost_sq.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            # --- Per-row LSAP margin ---
            cost_sq_det = cost_sq.detach()
            if N > 0:
                row_idx_t = torch.as_tensor(row_ind[:N], device=device, dtype=torch.long)
                col_idx_t = torch.as_tensor(col_ind[:N], device=device, dtype=torch.long)
                row_costs = cost_sq_det.index_select(0, row_idx_t)
                chosen_costs = row_costs.gather(1, col_idx_t.unsqueeze(-1)).squeeze(-1)
                inf_src = torch.full_like(col_idx_t.unsqueeze(-1), 0, dtype=row_costs.dtype)
                inf_src.fill_(float("inf"))
                masked = row_costs.scatter(1, col_idx_t.unsqueeze(-1), inf_src)
                if masked.shape[1] > 1:
                    second_costs, _ = masked.min(dim=-1)
                else:
                    second_costs = torch.full_like(chosen_costs, float("inf"))
                row_margins = second_costs - chosen_costs
            else:
                row_margins = torch.empty(0, device=device, dtype=slots.dtype)

            # --- Interpret assignment: two passes (CONTINUE/RE-ENTER, then BIRTH) ---
            output = torch.zeros_like(slots[b])
            source_for_identity = torch.full((N,), -1, device=device, dtype=torch.long)
            claimed_positions: set = set()
            birth_rows: List[int] = []
            new_dormant: List[Dict] = []
            matched_active: set = set()
            matched_dormant: set = set()
            out_pos_to_row: List[int] = [-1] * N
            reenter_candidates: List[Tuple[int, int]] = []

            # The reject test uses the UN-weighted appearance distance (cosine),
            # not the weighted cost, so that the tuning of ``reject_threshold``
            # matches :class:`MAPWithRejectPredictor`. ``cost[r, c]`` here is
            # weighted; we reconstruct the appearance term for the threshold
            # check below.
            for r, c in zip(row_ind[:N], col_ind[:N]):
                # Appearance distance at (r, c): available from cost matrix
                # (before depth / lambda). We recompute to be explicit.
                if c < n_active:
                    app_dist = (1.0 - cur_feats[r] @ active_feats[c]).item()
                elif c < n_cols:
                    # Dormant column: age penalty was added to appearance term.
                    app_dist = (
                        (1.0 - cur_feats[r] @ F.normalize(
                            torch.stack(
                                [d["feature"] for d in dormant_entries]
                            ).to(device), dim=-1
                        )[c - n_active]).item()
                    )
                else:
                    # Pure padding column → always a birth.
                    app_dist = self.reject_threshold + 1.0

                if app_dist > self.reject_threshold:
                    birth_rows.append(int(r))
                elif c < n_active:
                    output[c] = slots[b, r]
                    source_for_identity[c] = r
                    claimed_positions.add(c)
                    matched_active.add(c)
                    out_pos_to_row[c] = int(r)
                else:
                    d_idx = c - n_active
                    reenter_candidates.append((int(r), d_idx))

            for row_idx, d_idx in reenter_candidates:
                orig_idx = dormant_entries[d_idx].get("orig_idx", -1)
                if 0 <= orig_idx < N and orig_idx not in claimed_positions:
                    output[orig_idx] = slots[b, row_idx]
                    source_for_identity[orig_idx] = row_idx
                    claimed_positions.add(orig_idx)
                    matched_dormant.add(d_idx)
                    out_pos_to_row[orig_idx] = int(row_idx)
                else:
                    birth_rows.append(row_idx)

            unclaimed = sorted(set(range(N)) - claimed_positions)
            for row_idx, out_pos in zip(birth_rows, unclaimed):
                output[out_pos] = slots[b, row_idx]
                source_for_identity[out_pos] = row_idx
                claimed_positions.add(out_pos)
                out_pos_to_row[out_pos] = int(row_idx)
            if len(birth_rows) > len(unclaimed):
                import warnings
                warnings.warn(
                    f"DepthAugmentedMAPPredictor: {len(birth_rows)} births but "
                    f"only {len(unclaimed)} unclaimed positions; dropping "
                    f"{len(birth_rows) - len(unclaimed)} births"
                )
                self._last_births_overflow = True

            # --- DEATH: unmatched active → dormant (carry last-frame histogram) ---
            for j in range(n_active):
                if j not in matched_active:
                    entry: Dict[str, Any] = {
                        "feature": reference[b, j].detach().cpu(),
                        "age": 0,
                        "orig_idx": j,
                    }
                    if prev_hist is not None and b < prev_hist.shape[0]:
                        entry["hist"] = prev_hist[b, j].detach().cpu()
                    new_dormant.append(entry)

            # --- Age & evict dormant; carry matched entries forward ---
            for d_idx, d_entry in enumerate(dormant_entries):
                if d_idx not in matched_dormant:
                    d_entry["age"] += 1
                    if d_entry["age"] <= self.max_dormant_age:
                        new_dormant.append(d_entry)

            self._dormant[b] = new_dormant
            reordered_list.append(output)
            indices_list.append(source_for_identity)

            if N > 0:
                out_margins_b = torch.full(
                    (N,), float("inf"), device=device, dtype=slots.dtype
                )
                for out_pos, r in enumerate(out_pos_to_row):
                    if 0 <= r < N:
                        out_margins_b[out_pos] = row_margins[r]
                margin_list.append(out_margins_b)
            else:
                margin_list.append(row_margins)

            # --- Reorder current-frame histograms to match output positions ---
            if new_hist_by_slot is not None and cur_hist is not None:
                h_b = cur_hist[b]                                 # [N, bins]
                reordered_hist_b = torch.full_like(h_b, 1.0 / float(self.depth_hist_bins))
                for out_pos, r in enumerate(out_pos_to_row):
                    if 0 <= r < N:
                        reordered_hist_b[out_pos] = h_b[r]
                new_hist_by_slot.append(reordered_hist_b)

        reordered = torch.stack(reordered_list, dim=0)
        self._prev_slots = reordered.detach()
        self._last_match_indices = torch.stack(indices_list, dim=0)
        if len(margin_list) > 0:
            self._last_margin_per_slot = torch.stack(margin_list, dim=0).detach()
        else:
            self._last_margin_per_slot = torch.empty(
                (B, 0), device=device, dtype=slots.dtype
            )
        if new_hist_by_slot is not None:
            self._prev_hist_by_slot = torch.stack(new_hist_by_slot, dim=0).detach()
        elif not use_depth:
            # If depth became unavailable this frame but was available earlier,
            # drop the stale state so we re-initialize on the next depth-bearing
            # frame instead of mixing resolutions.
            self._prev_hist_by_slot = None

        if existence_mask is not None:
            reordered_mask = torch.stack(
                [existence_mask[b, indices_list[b]] for b in range(B)], dim=0
            )
            self._last_existence_mask = reordered_mask
            return reordered, reordered_mask, None
        self._last_existence_mask = None
        if return_weights:
            return reordered, None
        return reordered


class MaskIoUQualityHead(nn.Module):
    """Per-slot mask quality prediction head (Idea #023).

    A small 2-layer MLP that predicts a scalar quality score in ``[0, 1]`` for
    each slot. The score is trained to predict either

      - the slot's mask IoU with its ground-truth mask (Table B), or
      - the slot's decoder self-agreement IoU (Table A).

    The head is a secondary output head: it consumes the slot features and
    emits a per-slot quality. The quality can then be used downstream for
    loss reweighting or pseudo-label selection, but this head itself does NOT
    modify routing or any other module's behavior.

    Architecture::

        Linear(dim, hidden) -> GELU -> Linear(hidden, 1) -> sigmoid

    Args:
        dim: Slot feature dimension (input channel count).
        hidden: Hidden dimension of the 2-layer MLP. Defaults to 256.

    Forward:
        slots ``[B, N, D]`` -> quality ``[B, N]`` in ``[0, 1]``.
    """

    def __init__(self, dim: int, hidden: int = 256, **kwargs):
        super().__init__()
        del kwargs  # reserved for future configuration (e.g. dropout).
        self.dim = dim
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """Predict per-slot quality in ``[0, 1]``.

        Args:
            slots: ``[B, N, D]`` slot features.

        Returns:
            ``[B, N]`` quality scores in ``[0, 1]``.
        """
        logits = self.net(slots).squeeze(-1)  # [B, N]
        return torch.sigmoid(logits)


class OpenSetIdentityHead(nn.Module):
    """Open-set per-slot identity head for GSRS (proposal §3.1(a), §3.13).

    A 3-layer MLP that consumes a per-slot state and emits a distribution
    over the 4-simplex ``{source, dormant, null, born}``. The head is
    supervised by the replay-event flags produced by the slot-compositional
    renderer (§3.10). At inference the argmax over the 4 categories drives
    the unified variable-K + occlusion + re-ID decision rule (§3.13),
    replacing the hand-tuned thresholds that GCv1 uses.

    Architecture::

        Linear(dim, hidden) -> GELU ->
        Linear(hidden, hidden) -> GELU ->
        Linear(hidden, 4) -> softmax

    Input shape is preserved up to the final dim: the MLP is applied
    independently to every slot position, so inputs of shape
    ``[B, T, K, D]`` return ``[B, T, K, 4]``, and inputs of shape
    ``[B, K, D]`` return ``[B, K, 4]``.

    Gradients flow back through ``slots``: the caller is responsible for
    ensuring the upstream slot state has ``requires_grad=True`` when the
    head is being trained. We make no assumption beyond the standard
    :class:`torch.nn.Module` contract — in particular we do NOT declare
    ``requires_depth`` or any other cross-module signal contract, so the
    head drops into the existing ``networks.`` factory dispatch without
    disturbing sibling components.
    """

    # Enumerate the 4 categories emitted by the softmax head. The order is
    # part of the training contract: it must match ``GSRSIdentityLoss``'s
    # ``EVENT_TO_INDEX`` mapping. Documented here rather than hidden inside
    # a one-off comment so downstream consumers (inference script §3.13)
    # have a single source of truth.
    CATEGORIES: Tuple[str, ...] = ("source", "dormant", "null", "born")

    def __init__(
        self,
        dim: int,
        hidden: int = 256,
        **kwargs,
    ):
        super().__init__()
        del kwargs  # reserved for future extensions (e.g. dropout).
        if dim <= 0:
            raise ValueError(f"`dim` must be > 0, got {dim}.")
        if hidden <= 0:
            raise ValueError(f"`hidden` must be > 0, got {hidden}.")
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.num_categories = len(self.CATEGORIES)
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.num_categories),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """Predict per-slot open-set identity probabilities.

        Args:
            slots: ``[..., D]`` per-slot state tensor. Typical shapes are
                ``[B, T, K, D]`` (video, our primary case at
                ``D_slot = 128``) and ``[B, K, D]`` (image). The MLP is
                applied independently along the feature axis, so any
                leading batch-like dims are preserved.

        Returns:
            ``[..., 4]`` per-slot probabilities over ``CATEGORIES``,
            normalised along the final axis by a softmax.
        """
        if slots.ndim < 2:
            raise ValueError(
                "OpenSetIdentityHead expects `slots` with at least 2 dims "
                f"(last = feature dim); got ndim={slots.ndim}."
            )
        if slots.shape[-1] != self.dim:
            raise ValueError(
                f"OpenSetIdentityHead feature dim mismatch: expected {self.dim}, "
                f"got {slots.shape[-1]} (input shape {tuple(slots.shape)})."
            )
        logits = self.net(slots)  # [..., 4]
        return torch.softmax(logits, dim=-1)


class KalmanLSAPPredictor(nn.Module):
    """Kalman-augmented LSAP slot predictor (idea #019, KalmanLSAP).

    Wraps :class:`MAPWithRejectPredictor` with a per-slot constant-velocity
    Kalman filter on the 2-D centroid. At each frame we:

      1. Predict the next centroid + innovation covariance for every slot
         carried from ``t-1``.
      2. Augment the LSAP cost with a Mahalanobis motion term whose
         contribution is *uncertainty-aware*: when covariance is large
         (freshly born / lost track) the motion term auto-saturates toward
         a uniform prior and appearance dominates; when covariance is
         tight (well-tracked) the motion term has a sharp basin and
         dominates. This improves on v9's learned cross-attention rescue
         by explicitly modelling slot trajectories.
      3. Delegate the actual assignment to the inner MAP module (it owns
         the dormant registry + DEATH/BIRTH/RE-ENTER bookkeeping).
      4. Apply a Kalman update per matched identity using the new
         centroid.

    Design notes
    ------------
    * We do NOT modify :class:`MAPWithRejectPredictor`. Instead we
      (i) run the inner MAP to get its appearance-based
      ``source_for_identity`` mapping, (ii) compute the motion cost
      matrix from the previous-frame Kalman state (exposed as
      ``_last_motion_cost`` for ablations), and (iii) always return the
      MAP-reordered slots. The Kalman state is updated using that
      assignment; when ``lambda_motion == 0`` we reduce exactly to
      :class:`MAPWithRejectPredictor`.

    * ``slot_centroids`` arrives in normalized image coords. The task
      intent specifies ``[-1, 1]``; upstream code in ``video.py`` currently
      produces ``(h, w)`` normalized to ``[0, 1]``. Either convention
      works because the Kalman filter is linear in centroid units — we
      do NOT rescale.

    * State is MUTABLE per batch and fully detached (no gradient flows
      through the Kalman update). ``reset()`` clears it, mirroring
      :class:`NullAwareMemoryPredictor`.

    * Graceful fallback: if ``slot_centroids`` is ``None`` we delegate to
      the inner MAP verbatim and return its output unchanged.
    """

    def __init__(
        self,
        dim: int,
        similarity: str = "cosine",
        reject_threshold: float = 0.5,
        age_penalty: float = 0.05,
        max_dormant_age: int = 10,
        lambda_appearance: float = 1.0,
        lambda_motion: float = 0.3,
        kalman_process_noise: float = 0.03,
        kalman_obs_noise: float = 0.05,
        pre_match: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.similarity = similarity
        self.lambda_appearance = float(lambda_appearance)
        self.lambda_motion = float(lambda_motion)
        self.kalman_process_noise = float(kalman_process_noise)
        self.kalman_obs_noise = float(kalman_obs_noise)
        self.pre_match = bool(pre_match)

        # Inner MAP-with-reject predictor; owns appearance matching +
        # dormant registry + DEATH/BIRTH/RE-ENTER logic.
        self.map = MAPWithRejectPredictor(
            dim=dim,
            similarity=similarity,
            max_dormant_age=max_dormant_age,
            reject_threshold=reject_threshold,
            age_penalty=age_penalty,
            pre_match=pre_match,
        )

        # Sentinel for LatentProcessor dispatch (video.py checks hasattr).
        self._hungarian_match = None
        # Trigger centroid plumbing in video.py (caller computes per-slot
        # centroids from the current-frame masks before calling us).
        self.use_hybrid_cost = True

        # --- Per-slot Kalman state (per-batch lists, mutable, detached) ---
        # ``self._kalman_states[b][c]`` is a dict
        # ``{"mu": [4] tensor, "cov": [4, 4] tensor}`` where the state is
        # ``[x, y, vx, vy]`` (or equivalently ``[u, v, vu, vv]``; the two
        # spatial axes are treated symmetrically). ``None`` means no state
        # yet for that identity slot.
        self._kalman_states: Optional[List[List[Optional[Dict[str, torch.Tensor]]]]] = None
        # Diagnostic: last motion cost matrix per batch element (detached).
        self._last_motion_cost: Optional[List[Optional[torch.Tensor]]] = None

        # Mirror MAP's persisted state for callers that read this module
        # directly (e.g. a HybridPredictor-style wrapper).
        self._prev_slots: Optional[torch.Tensor] = None
        self._last_match_indices: Optional[torch.Tensor] = None
        self._last_existence_mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ utils
    def reset(self) -> None:
        """Reset Kalman state AND the inner MAP state."""
        self.map.reset()
        self._kalman_states = None
        self._last_motion_cost = None
        self._prev_slots = None
        self._last_match_indices = None
        self._last_existence_mask = None

    def get_last_match_indices(self) -> Optional[torch.Tensor]:
        return self.map.get_last_match_indices()

    def match_to_reference(self, slots: torch.Tensor) -> torch.Tensor:
        """V1 interface for pre-matching mode."""
        out = self.forward(slots)
        return out if isinstance(out, torch.Tensor) else out[0]

    # ------------------------------------------------------------- Kalman ops
    @staticmethod
    def _kalman_F(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Constant-velocity state transition ``F ∈ R^{4x4}``."""
        F_mat = torch.eye(4, device=device, dtype=dtype)
        F_mat[0, 2] = 1.0  # x += vx
        F_mat[1, 3] = 1.0  # y += vy
        return F_mat

    @staticmethod
    def _kalman_H(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Observation matrix ``H ∈ R^{2x4}`` — observe ``(x, y)`` only."""
        H_mat = torch.zeros(2, 4, device=device, dtype=dtype)
        H_mat[0, 0] = 1.0
        H_mat[1, 1] = 1.0
        return H_mat

    def _kalman_Q(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Process noise covariance ``Q = σ_q^2 · I_4``."""
        q2 = self.kalman_process_noise ** 2
        return torch.eye(4, device=device, dtype=dtype) * q2

    def _kalman_R(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Observation noise covariance ``R = σ_r^2 · I_2``."""
        r2 = self.kalman_obs_noise ** 2
        return torch.eye(2, device=device, dtype=dtype) * r2

    def _kalman_init(self, centroid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Initialize a Kalman state from a single 2-D centroid observation.

        Position is set to the observation; velocity is zero; covariance
        is ``R`` on position (matches obs noise) and a large finite value
        on velocity (we don't know it yet).
        """
        device = centroid.device
        dtype = centroid.dtype
        mu = torch.zeros(4, device=device, dtype=dtype)
        mu[0] = centroid[0]
        mu[1] = centroid[1]
        cov = torch.eye(4, device=device, dtype=dtype)
        r2 = self.kalman_obs_noise ** 2
        cov[0, 0] = r2
        cov[1, 1] = r2
        # Velocity uncertainty: large but finite. ``1.0`` in normalized
        # coordinates corresponds to roughly "half an image per frame",
        # which is very permissive — exactly what we want at birth.
        cov[2, 2] = 1.0
        cov[3, 3] = 1.0
        return {"mu": mu.detach(), "cov": cov.detach()}

    def _kalman_predict(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-step predict. Returns ``(mu_pred, cov_pred, innov_cov_S)``."""
        mu = state["mu"]
        cov = state["cov"]
        device = mu.device
        dtype = mu.dtype
        F_mat = self._kalman_F(device, dtype)
        Q_mat = self._kalman_Q(device, dtype)
        H_mat = self._kalman_H(device, dtype)
        R_mat = self._kalman_R(device, dtype)
        mu_pred = F_mat @ mu
        cov_pred = F_mat @ cov @ F_mat.T + Q_mat
        # Innovation covariance S = H P_pred H^T + R ∈ R^{2x2}
        S = H_mat @ cov_pred @ H_mat.T + R_mat
        return mu_pred.detach(), cov_pred.detach(), S.detach()

    def _kalman_update(
        self,
        state: Dict[str, torch.Tensor],
        centroid_obs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict + measurement update with observation ``centroid_obs``."""
        mu_pred, cov_pred, S = self._kalman_predict(state)
        device = mu_pred.device
        dtype = mu_pred.dtype
        H_mat = self._kalman_H(device, dtype)
        # Kalman gain K = P_pred H^T S^{-1} ∈ R^{4x2}
        try:
            S_inv = torch.linalg.inv(S)
        except RuntimeError:
            S_reg = S + 1e-6 * torch.eye(2, device=device, dtype=dtype)
            S_inv = torch.linalg.inv(S_reg)
        K = cov_pred @ H_mat.T @ S_inv
        innov = centroid_obs - H_mat @ mu_pred  # [2]
        mu_new = mu_pred + K @ innov
        I4 = torch.eye(4, device=device, dtype=dtype)
        cov_new = (I4 - K @ H_mat) @ cov_pred
        return {"mu": mu_new.detach(), "cov": cov_new.detach()}

    @staticmethod
    def _mahalanobis_sq(
        obs: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
        """Squared Mahalanobis distance ``(o - m)^T Σ^{-1} (o - m)``.

        Returns a scalar tensor. ``cov`` is regularized before inversion
        so we stay numerically stable on near-singular matrices.
        """
        device = obs.device
        dtype = obs.dtype
        diff = (obs - mean).to(device=device, dtype=dtype)
        reg = 1e-6 * torch.eye(cov.shape[-1], device=device, dtype=dtype)
        try:
            inv = torch.linalg.inv(cov + reg)
        except RuntimeError:
            inv = torch.eye(cov.shape[-1], device=device, dtype=dtype)
        return diff @ inv @ diff

    # --------------------------------------------------------------- forward
    def forward(
        self,
        slots: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        slot_centroids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        """Kalman-augmented LSAP match.

        This method re-implements the three-pass LSAP logic from
        :class:`MAPWithRejectPredictor.forward` (networks.py lines
        2171-2443) and augments the active-block appearance cost with a
        Kalman-Mahalanobis motion cost so that the assignment is driven
        by BOTH appearance AND motion, not appearance alone. The dormant
        registry state (``_dormant``, ``_next_id``) is still owned by the
        inner :class:`MAPWithRejectPredictor` instance (``self.map``) so
        that DEATH / RE-ENTER bookkeeping remains bit-identical to the
        MAP implementation.

        Args:
            slots: ``[B, N, D]`` current-frame slots.
            prev_slots: optional override for cached previous slots
                (mirrors the :class:`MAPWithRejectPredictor` contract).
            existence_mask: ``[B, N]`` alive flag.
            features: unused (forwarded for API parity).
            slot_centroids: ``[B, N, 2]`` slot centroid in normalized
                image coords. If ``None``, we fall back to pure MAP
                matching (delegated to ``self.map.forward``).
            return_weights: attention-weight compatibility; always
                returns ``None``.

        Returns:
            Mirrors :class:`MAPWithRejectPredictor`. When
            ``existence_mask`` is supplied, returns
            ``(reordered, reordered_mask, None)``; otherwise returns
            ``reordered`` (or ``(reordered, None)`` if ``return_weights``
            is truthy).
        """
        from scipy.optimize import linear_sum_assignment

        # ``video.py`` dispatches centroids under the keyword ``centroids``;
        # accept both spellings.
        if slot_centroids is None:
            slot_centroids = kwargs.pop("centroids", None)
        else:
            kwargs.pop("centroids", None)

        B, N, _D = slots.shape
        device = slots.device
        dtype = slots.dtype

        # Graceful fallback: no centroids → pure MAP behaviour. This also
        # keeps us backwards-compatible with any upstream that has not yet
        # wired the centroid plumbing.
        if slot_centroids is None:
            map_out = self.map.forward(
                slots,
                prev_slots=prev_slots,
                existence_mask=existence_mask,
                return_weights=return_weights,
            )
            # Mirror MAP's persisted state so external callers see it on us.
            self._prev_slots = self.map._prev_slots
            self._last_match_indices = self.map.get_last_match_indices()
            self._last_existence_mask = self.map._last_existence_mask
            self._last_motion_cost = [None] * B
            return map_out

        reference = prev_slots if prev_slots is not None else self.map._prev_slots

        # -------------------------------------------------- first frame case
        # Copy of MAPWithReject first-frame branch (networks.py lines
        # 2194-2209) with the addition of Kalman state initialization.
        if reference is None:
            self.map._prev_slots = slots.detach()
            self.map._dormant = [[] for _ in range(B)]
            self.map._next_id = [N for _ in range(B)]
            self.map._last_match_indices = None
            self.map._last_margin_per_slot = torch.full(
                (B, N), float("inf"), device=device, dtype=dtype
            )
            self.map._last_births_overflow = False
            self.map._last_existence_mask = existence_mask

            # Initialize per-slot Kalman state from the first-frame observations.
            self._kalman_states = [[None] * N for _ in range(B)]
            cent_det = slot_centroids.detach().to(device=device, dtype=dtype)
            for b in range(B):
                for c in range(N):
                    if existence_mask is not None and not bool(
                        existence_mask[b, c].item()
                    ):
                        continue
                    self._kalman_states[b][c] = self._kalman_init(cent_det[b, c])

            self._last_motion_cost = [None] * B
            self._prev_slots = slots.detach()
            self._last_match_indices = None
            self._last_existence_mask = existence_mask

            if existence_mask is not None:
                return slots, existence_mask, None
            return slots if not return_weights else (slots, None)

        # --- Ensure inner MAP state invariants (dormant / next_id) exist.
        if self.map._dormant is None:
            self.map._dormant = [[] for _ in range(B)]
        if self.map._next_id is None:
            self.map._next_id = [N for _ in range(B)]

        # Normalize for cosine distance (copy of MAPWithReject L2212-2213).
        ref_norm = F.normalize(reference, dim=-1)   # [B, N, D]
        cur_norm = F.normalize(slots, dim=-1)        # [B, N, D]

        cent_det = slot_centroids.detach().to(device=device, dtype=dtype)
        if self._kalman_states is None or len(self._kalman_states) != B:
            self._kalman_states = [[None] * N for _ in range(B)]

        reordered_list = []
        indices_list = []
        margin_list = []
        motion_cost_list: List[Optional[torch.Tensor]] = [None] * B
        self.map._last_births_overflow = False

        # ================================================================
        # BEGIN copy of MAPWithRejectPredictor.forward (networks.py lines
        # 2222-2420), with the ACTIVE-block cost augmented by the Kalman
        # Mahalanobis motion term:
        #   cost_active = lambda_appearance * (1 - cur @ active.T)
        #               + lambda_motion     * mahalanobis_sq(cur_centroid,
        #                                       predicted_centroid; S)
        # The dormant block stays appearance-cosine-only because we do
        # not track motion state for dormant entries (their positions
        # from n-frames ago are stale); when an identity re-enters we
        # re-seed its Kalman state from the observed centroid.
        # ================================================================
        for b in range(B):
            active_feats = ref_norm[b]   # [N, D]
            dormant_entries = self.map._dormant[b]

            n_active = N
            n_dormant = len(dormant_entries)
            n_cols = n_active + n_dormant

            cur_feats = cur_norm[b]  # [N, D]

            # --- Appearance cost (copy of MAPWithReject L2238).
            active_cost_app = 1.0 - (cur_feats @ active_feats.T)  # [N, N]

            # --- Motion cost from per-slot Kalman prediction. Columns
            # correspond to prev-frame identity slots (same ordering as
            # ``active_feats`` / ``reference[b]``), rows to current slots.
            # Slots whose Kalman state is ``None`` (never tracked, or
            # dropped after DEATH / BIRTH) contribute zero motion cost
            # and thus let appearance alone decide — the innovation
            # covariance is also unavailable in that case.
            motion_cost = torch.zeros(N, N, device=device, dtype=dtype)
            kf_states_b = self._kalman_states[b] if b < len(self._kalman_states) else None
            if kf_states_b is not None and len(kf_states_b) == N:
                pred_centroids = torch.zeros(N, 2, device=device, dtype=dtype)
                innov_covs = torch.zeros(N, 2, 2, device=device, dtype=dtype)
                valid = torch.zeros(N, device=device, dtype=torch.bool)
                for c in range(N):
                    st = kf_states_b[c]
                    if st is None:
                        continue
                    mu_pred, _cov_pred, S = self._kalman_predict(st)
                    pred_centroids[c, 0] = mu_pred[0]
                    pred_centroids[c, 1] = mu_pred[1]
                    innov_covs[c] = S
                    valid[c] = True
                for r in range(N):
                    for c in range(N):
                        if not bool(valid[c].item()):
                            continue
                        motion_cost[r, c] = self._mahalanobis_sq(
                            cent_det[b, r], pred_centroids[c], innov_covs[c]
                        )

            # --- Combined active-block cost (THE AUGMENTATION). When
            # ``lambda_motion == 0`` and ``lambda_appearance == 1`` this
            # reduces exactly to MAP's appearance-only cost.
            active_cost = (
                self.lambda_appearance * active_cost_app
                + self.lambda_motion * motion_cost
            )
            motion_cost_list[b] = (self.lambda_motion * motion_cost).detach()

            # --- Copy of MAPWithReject L2242-2246: mask invalid slots.
            if existence_mask is not None:
                cur_invalid = ~existence_mask[b].bool()
                prev_invalid = cur_invalid
                active_cost[cur_invalid, :] = 1e6
                active_cost[:, prev_invalid] = 1e6

            # --- Copy of MAPWithReject L2248-2263: dormant block
            # (appearance-cosine + age penalty; motion not tracked for
            # dormant entries — see docstring).
            if n_dormant > 0:
                dormant_feats = torch.stack(
                    [d["feature"] for d in dormant_entries]
                ).to(device)  # [n_dormant, D]
                dormant_feats = F.normalize(dormant_feats, dim=-1)
                dormant_cost = 1.0 - (cur_feats @ dormant_feats.T)
                ages = torch.tensor(
                    [d["age"] for d in dormant_entries],
                    device=device, dtype=torch.float32
                )
                dormant_cost = dormant_cost + self.map.age_penalty * ages.unsqueeze(0)
                cost = torch.cat([active_cost, dormant_cost], dim=1)
            else:
                cost = active_cost

            # --- Copy of MAPWithReject L2266-2277: pad to square for LSAP.
            n_rows = N
            if n_cols > n_rows:
                pad = torch.full((n_cols - n_rows, n_cols), 1e6, device=device)
                cost_sq = torch.cat([cost, pad], dim=0)
            elif n_rows > n_cols:
                pad = torch.full(
                    (n_rows, n_rows - n_cols),
                    self.map.reject_threshold + 0.01,
                    device=device,
                )
                cost_sq = torch.cat([cost, pad], dim=1)
            else:
                cost_sq = cost

            # --- Copy of MAPWithReject L2279-2281: solve LSAP on the
            # augmented cost (appearance + motion).
            cost_np = cost_sq.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            # --- Copy of MAPWithReject L2283-2313: per-row ambiguity margin.
            cost_sq_det = cost_sq.detach()
            if N > 0:
                row_idx_t = torch.as_tensor(row_ind[:N], device=device, dtype=torch.long)
                col_idx_t = torch.as_tensor(col_ind[:N], device=device, dtype=torch.long)
                row_costs = cost_sq_det.index_select(0, row_idx_t)
                chosen_costs = row_costs.gather(1, col_idx_t.unsqueeze(-1)).squeeze(-1)
                inf_src = torch.full_like(col_idx_t.unsqueeze(-1), 0, dtype=row_costs.dtype)
                inf_src.fill_(float("inf"))
                masked = row_costs.scatter(1, col_idx_t.unsqueeze(-1), inf_src)
                if masked.shape[1] > 1:
                    second_costs, _ = masked.min(dim=-1)
                else:
                    second_costs = torch.full_like(chosen_costs, float("inf"))
                row_margins = second_costs - chosen_costs
            else:
                row_margins = torch.empty(0, device=device, dtype=dtype)

            # --- Copy of MAPWithReject L2315-2374: three-pass CONTINUE /
            # RE-ENTER / BIRTH interpretation of the assignment.
            output = torch.zeros_like(slots[b])
            source_for_identity = torch.full((N,), -1, device=device, dtype=torch.long)
            claimed_positions = set()
            birth_rows: List[int] = []
            new_dormant: List[Dict[str, Any]] = []
            matched_active: set = set()
            matched_dormant: set = set()
            out_pos_to_row: List[int] = [-1] * N

            # Pass 1a: CONTINUE vs defer RE-ENTER.
            reenter_candidates: List[Tuple[int, int]] = []
            for r, c in zip(row_ind[:N], col_ind[:N]):
                actual_cost = (
                    cost[r, c].item()
                    if c < n_cols
                    else self.map.reject_threshold + 1
                )
                if actual_cost > self.map.reject_threshold:
                    birth_rows.append(int(r))
                elif c < n_active:
                    output[c] = slots[b, r]
                    source_for_identity[c] = r
                    claimed_positions.add(c)
                    matched_active.add(c)
                    out_pos_to_row[c] = int(r)
                else:
                    d_idx = c - n_active
                    reenter_candidates.append((int(r), d_idx))

            # Pass 1b: RE-ENTER.
            for row_idx, d_idx in reenter_candidates:
                orig_idx = dormant_entries[d_idx].get("orig_idx", -1)
                if 0 <= orig_idx < N and orig_idx not in claimed_positions:
                    output[orig_idx] = slots[b, row_idx]
                    source_for_identity[orig_idx] = row_idx
                    claimed_positions.add(orig_idx)
                    matched_dormant.add(d_idx)
                    out_pos_to_row[orig_idx] = int(row_idx)
                else:
                    birth_rows.append(row_idx)

            # Pass 2: BIRTH.
            unclaimed = sorted(set(range(N)) - claimed_positions)
            for row_idx, out_pos in zip(birth_rows, unclaimed):
                output[out_pos] = slots[b, row_idx]
                source_for_identity[out_pos] = row_idx
                claimed_positions.add(out_pos)
                out_pos_to_row[out_pos] = int(row_idx)
            if len(birth_rows) > len(unclaimed):
                import warnings
                warnings.warn(
                    f"KalmanLSAPPredictor: {len(birth_rows)} births but only "
                    f"{len(unclaimed)} unclaimed positions; dropping "
                    f"{len(birth_rows) - len(unclaimed)} births"
                )
                self.map._last_births_overflow = True

            # --- Copy of MAPWithReject L2386-2403: DEATH → push to dormant,
            # age+evict existing dormant entries.
            for j in range(n_active):
                if j not in matched_active:
                    new_dormant.append({
                        "feature": reference[b, j].detach().cpu(),
                        "age": 0,
                        "orig_idx": j,
                    })
            for d_idx, entry in enumerate(dormant_entries):
                if d_idx not in matched_dormant:
                    entry["age"] += 1
                    if entry["age"] <= self.map.max_dormant_age:
                        new_dormant.append(entry)

            self.map._dormant[b] = new_dormant
            reordered_list.append(output)
            indices_list.append(source_for_identity)

            # --- Copy of MAPWithReject L2407-2420: project row margins to
            # output-slot margins.
            if N > 0:
                out_margins_b = torch.full(
                    (N,), float("inf"), device=device, dtype=dtype
                )
                for out_pos, r in enumerate(out_pos_to_row):
                    if 0 <= r < N:
                        out_margins_b[out_pos] = row_margins[r]
                margin_list.append(out_margins_b)
            else:
                margin_list.append(row_margins)

            # --- Kalman state update for this batch element. Output slot
            # ``c`` now carries identity ``c`` (by construction of the
            # three-pass logic); we update its Kalman state using the
            # current-frame centroid observed at row ``out_pos_to_row[c]``.
            # CONTINUE → true predict+update; RE-ENTER / BIRTH → re-init
            # from observation (no usable motion history).
            new_states: List[Optional[Dict[str, torch.Tensor]]] = [None] * N
            prev_states = self._kalman_states[b]
            for c in range(N):
                r = out_pos_to_row[c]
                if r < 0 or r >= N:
                    new_states[c] = None
                    continue
                if existence_mask is not None and not bool(
                    existence_mask[b, r].item()
                ):
                    new_states[c] = None
                    continue
                obs = cent_det[b, r]
                if c in matched_active and prev_states[c] is not None:
                    new_states[c] = self._kalman_update(prev_states[c], obs)
                else:
                    # RE-ENTER (dormant motion state lost) or BIRTH: seed fresh.
                    new_states[c] = self._kalman_init(obs)
            self._kalman_states[b] = new_states

        # ================================================================
        # END copy of MAPWithRejectPredictor.forward body.
        # ================================================================

        reordered = torch.stack(reordered_list, dim=0)  # [B, N, D]
        last_match = torch.stack(indices_list, dim=0)    # [B, N]

        # --- Persist state on the inner MAP so its diagnostics / next-frame
        # invariants stay consistent (copy of MAPWithReject L2422-2432).
        self.map._prev_slots = reordered.detach()
        self.map._last_match_indices = last_match
        if len(margin_list) > 0:
            self.map._last_margin_per_slot = torch.stack(margin_list, dim=0).detach()
        else:
            self.map._last_margin_per_slot = torch.empty(
                (B, 0), device=device, dtype=dtype
            )

        # --- Build per-output existence mask (copy of MAPWithReject L2434-2440).
        if existence_mask is not None:
            reordered_mask = torch.stack(
                [existence_mask[b, indices_list[b]] for b in range(B)], dim=0
            )
            self.map._last_existence_mask = reordered_mask
        else:
            reordered_mask = None
            self.map._last_existence_mask = None

        # --- Mirror MAP's persisted state on the wrapper too.
        self._prev_slots = reordered.detach()
        self._last_match_indices = last_match
        self._last_existence_mask = reordered_mask
        self._last_motion_cost = motion_cost_list

        # --- Return contract (mirrors MAPWithRejectPredictor).
        if existence_mask is not None:
            return reordered, reordered_mask, None
        if return_weights:
            return reordered, None
        return reordered


class CutieSlotPredictor(HungarianPredictor):
    """Hungarian predictor augmented with a Cutie-style object-query memory bank.

    Hypothesis H1 (cross-field synthesis 2026-04-16 §3): Giving each slot a
    persistent long-term identity memory — the Cutie object-query primitive
    (Cheng et al. 2024, arXiv:2310.12982) — lets the predictor rescue identity
    through multi-frame occlusion events without any learned parameters. This
    is the ``𝓕_mem`` escape route from the χ-ceiling (see
    ``idea-stage/CROSS_FIELD_SYNTHESIS_2026_04_16.md`` §4 Escape 1): unlike
    ``𝓕_slot`` predictors whose input is only ``S_t``, ``CutieSlotPredictor``
    sees ``(S_t, M_t)`` where ``M_t`` is the read of past-frame slot features.

    Mechanism (spec §Module 2):
      1. Compute the standard Hungarian appearance cost
         ``C[i, j] = 0.5 * (1 - cos(S_prev[i], S_curr[j]))``.
      2. Query the memory bank for ``S_curr`` via ``bank.read`` → top-K memory
         entries per query slot.
      3. Compute the memory cost ``C_mem[i, j] = min_k 0.5 * (1 - cos(M[j][k], S_prev[i]))``;
         the min across top-K picks the best memory candidate that matches
         ``S_prev[i]``, so a slot at j that matches the IDENTITY of previous
         slot i gets a low memory cost even if its current appearance drifted.
      4. ``C_final[i, j] = min(C[i, j], λ_mem * C_mem[i, j] + penalty_nomatch)``
         — elementwise min means memory is a RESCUE channel, never a burden;
         pristine direct matches keep the standard cost.
      5. Solve Hungarian on ``C_final``; then write the matched
         (reordered) ``S_curr`` back to the bank tagged with timestamp ``t``.

    No silent fallback policy:
      - When the bank is empty (first frame / clip start), ``read`` returns
        ``valid=False`` for every entry. In that case ``C_mem`` is filled
        with ``+inf`` so the elementwise min cleanly degenerates to the
        standard Hungarian cost — documented, intentional, no hidden
        defaults.

    This class does NOT modify ``HungarianPredictor`` in place (it subclasses
    it per spec). All memory-bank interactions are confined to ``forward`` and
    ``reset``; ``_hungarian_match`` and ``_compute_hybrid_cost`` are inherited
    unchanged so the identity dispatch in ``video.py`` still triggers
    (checks ``hasattr(predictor, '_hungarian_match')``).

    Args:
        dim: Slot dimension (passed through to ``HungarianPredictor``).
        memory_capacity: Bank FIFO capacity per batch element. Default 64.
        top_k: Number of memory neighbours retrieved per query. Default 3.
        lambda_mem: Weight on the memory cost in the ``min(C, λ·C_mem + pen)``
            rule. Larger → memory competes with direct match sooner.
        penalty_nomatch: Additive penalty on ``λ·C_mem`` before the
            elementwise min. This prevents memory from winning the argmin
            unless the direct cost is clearly worse than memory cost +
            penalty — the ``log 2`` equivalent of a Bayes-factor barrier.
        similarity: Slot-to-slot similarity metric (inherited; only
            ``'cosine'`` is supported by the bank).
        **kwargs: Passed through to ``HungarianPredictor`` (e.g. ``pre_match``,
            ``use_hybrid_cost``, …). Hybrid cost is compatible but its cost is
            normalized differently; the memory path only touches the
            appearance channel.
    """

    def __init__(
        self,
        dim: int,
        memory_capacity: int = 64,
        top_k: int = 3,
        lambda_mem: float = 0.5,
        penalty_nomatch: float = 1.0,
        similarity: str = "cosine",
        **kwargs,
    ):
        if similarity != "cosine":
            raise NotImplementedError(
                "CutieSlotPredictor only supports similarity='cosine' (memory "
                f"bank retrieval requires it); got '{similarity}'."
            )
        super().__init__(dim=dim, similarity=similarity, **kwargs)
        # Local import to avoid circular imports on module init.
        from slotcontrast.modules.memory_bank import ObjectQueryMemoryBank

        self.memory_capacity = int(memory_capacity)
        self.top_k = int(top_k)
        self.lambda_mem = float(lambda_mem)
        self.penalty_nomatch = float(penalty_nomatch)

        self.memory_bank = ObjectQueryMemoryBank(
            capacity=self.memory_capacity,
            top_k=self.top_k,
            similarity=similarity,
        )
        # Per-clip frame counter. Incremented each forward once the bank has
        # been reset via ``ScanOverTime.forward``'s hook.
        self._frame_idx: int = 0
        # Diagnostics (last forward): whether memory actually lowered the
        # combined cost for any pair, and the fraction of pairs it helped.
        self._last_memory_helped_frac: float = 0.0

    def reset(self):
        """Reset predictor + bank state for a new video clip.

        Note: the bank needs ``batch_size`` to allocate its ragged buffers; we
        defer that allocation to ``ScanOverTime.forward``'s explicit
        ``memory_bank.reset(batch_size)`` hook. This ``reset`` only flushes
        the parent's matching state plus our frame counter. Calling ``reset``
        alone without the bank hook leaves the bank buffers in whatever size
        the previous clip used — the shape check inside ``bank.write`` /
        ``bank.read`` will raise loudly if misused.
        """
        super().reset()
        self._frame_idx = 0
        self._last_memory_helped_frac = 0.0
        # We intentionally do NOT call self.memory_bank.reset() here because
        # it requires batch_size. The ScanOverTime hook handles that.

    @property
    def last_memory_helped_frac(self) -> float:
        """Fraction of (i, j) pairs for which memory cost < direct cost.

        Diagnostic only — not used for gradients. A value near 0 means memory
        never helps (either bank empty, or direct matches are all stronger);
        a value near 1 would indicate memory dominates (potentially worth
        auditing for over-weighting ``lambda_mem``).
        """
        return float(self._last_memory_helped_frac)

    def _compute_memory_cost(
        self,
        prev_slots: torch.Tensor,
        curr_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the memory-augmented cost matrix ``C_mem[B, N, N]``.

        ``C_mem[b, i, j] = min_k 0.5 * (1 - cos(memory_reads[b, j, k], prev_slots[b, i]))``

        where ``memory_reads[b, j]`` are the top-k entries retrieved for
        ``curr_slots[b, j]``. Pads (no memory available) contribute ``+inf``.
        """
        B, N, _ = curr_slots.shape
        device = curr_slots.device
        dtype = curr_slots.dtype

        # Read top-k memory entries for each current slot.
        mem_feats, _mem_ts, mem_valid = self.memory_bank.read(
            curr_slots, top_k=self.top_k
        )
        # mem_feats:  [B, N, K, D]
        # mem_valid:  [B, N, K]

        K = mem_feats.shape[2]

        # Normalize prev and mem for cosine similarity.
        prev_norm = F.normalize(prev_slots, dim=-1)  # [B, N, D]
        mem_norm = F.normalize(mem_feats, dim=-1)  # [B, N, K, D]

        # Pairwise cosine similarity between prev[i] and mem[j, k]:
        # [B, 1, N, 1, D] * [B, N, 1, K, D] -> [B, N (j), N (i), K]
        # Rearranged via einsum for clarity.
        # cos_sim[b, i, j, k] = <prev_norm[b, i, :], mem_norm[b, j, k, :]>
        cos_sim = torch.einsum("bid,bjkd->bijk", prev_norm, mem_norm)
        # Cost in [0, 1]: 0.5 * (1 - cos_sim).
        costs = 0.5 * (1.0 - cos_sim)  # [B, N, N, K]

        # Mask invalid slots with +inf cost so they cannot win the min.
        # valid_mask: [B, N (j), K] -> broadcast to [B, N (i), N (j), K] via unsqueeze at dim=1.
        valid_mask = mem_valid.unsqueeze(1).expand(B, N, N, K)
        inf_fill = torch.full_like(costs, float("inf"))
        costs = torch.where(valid_mask, costs, inf_fill)

        # Reduce over top-k with min: C_mem[b, i, j] = min_k costs[b, i, j, k]
        c_mem, _ = costs.min(dim=-1)  # [B, N, N]

        return c_mem

    def forward(
        self,
        slots: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        centroids: Optional[torch.Tensor] = None,
        prev_centroids: Optional[torch.Tensor] = None,
        prev_prev_centroids: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        prev_masks: Optional[torch.Tensor] = None,
        attention_mass: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with memory-augmented Hungarian matching.

        Extra args vs parent:
            attention_mass: ``[B, N]`` per-slot attention mass used as the
                bank's eviction tie-breaker. If ``None``, attention mass
                defaults to uniform 1.0 (pure FIFO). No silent recomputation.
        """
        # pre_match modes mirror the parent class.
        if self.pre_match == "greedy":
            if existence_mask is not None:
                return slots, existence_mask, None
            if return_weights:
                return slots, None
            return slots

        if self.pre_match:
            self._prev_slots = slots.detach()
            if existence_mask is not None:
                return slots, existence_mask, None
            if return_weights:
                return slots, None
            return slots

        # Post-match path with memory augmentation.
        reference_slots = prev_slots if prev_slots is not None else self._prev_slots

        if reference_slots is None:
            # First frame of clip: no previous slots, no match to compute.
            # Write the initial slots into the bank so subsequent frames have
            # something to retrieve.
            self._prev_slots = slots.detach()
            self._last_match_indices = None
            self.memory_bank.write(slots, attention_mass, t=self._frame_idx)
            self._frame_idx += 1
            if existence_mask is not None:
                return slots, existence_mask, None
            if return_weights:
                return slots, None
            return slots

        # 1. Standard appearance cost (normalized to [0, 1]).
        prev_norm = F.normalize(reference_slots, dim=-1)
        curr_norm = F.normalize(slots, dim=-1)
        sim_matrix = torch.bmm(prev_norm, curr_norm.transpose(1, 2))
        direct_cost = 0.5 * (1.0 - sim_matrix)  # [B, N, N]

        # 2. Memory cost (may be +inf if bank empty).
        c_mem = self._compute_memory_cost(reference_slots, slots)  # [B, N, N]

        # 3. Combined cost: elementwise min(direct, λ·mem + penalty).
        mem_cost_scaled = self.lambda_mem * c_mem + self.penalty_nomatch
        combined_cost = torch.minimum(direct_cost, mem_cost_scaled)

        # Diagnostic: fraction of entries where memory strictly improved.
        # (inf comparisons safely evaluate to False.)
        helped = (mem_cost_scaled < direct_cost).float().mean().item()
        self._last_memory_helped_frac = float(helped)

        # 4. Hungarian on combined cost.
        from scipy.optimize import linear_sum_assignment

        B, N, _ = slots.shape
        device = slots.device
        reordered_list: List[torch.Tensor] = []
        indices_list: List[torch.Tensor] = []
        for b in range(B):
            cost_np = combined_cost[b].detach().cpu().numpy()
            _, col_ind = linear_sum_assignment(cost_np)
            reordered_list.append(slots[b, col_ind])
            indices_list.append(torch.tensor(col_ind, device=device, dtype=torch.long))
        reordered_slots = torch.stack(reordered_list, dim=0)
        indices = torch.stack(indices_list, dim=0)

        self._prev_slots = reordered_slots.detach()
        self._last_match_indices = indices

        # 5. Write matched slots back to the bank.
        # Reorder attention_mass to align with reordered slots.
        if attention_mass is not None:
            am_reordered = torch.stack(
                [attention_mass[b, indices[b]] for b in range(B)], dim=0
            )
        else:
            am_reordered = None
        self.memory_bank.write(reordered_slots, am_reordered, t=self._frame_idx)
        self._frame_idx += 1

        # Reorder existence mask if provided.
        if existence_mask is not None:
            reordered_mask = torch.stack(
                [existence_mask[b, indices[b]] for b in range(B)], dim=0
            )
            return reordered_slots, reordered_mask, None
        if return_weights:
            return reordered_slots, None
        return reordered_slots
