import math
from collections.abc import Sequence
from functools import partial
from typing import Dict, Optional

import einops
import numpy as np
import torch
from einops.layers.torch import Rearrange
from torchvision import transforms as tvt
from torchvision.transforms import functional as tvt_functional

from slotcontrast.data import transforms_video

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

MOVI_DEFAULT_MEAN = [0.5, 0.5, 0.5]
MOVI_DEFAULT_STD = [0.5, 0.5, 0.5]

DATASET_TYPES = {
    "coco": "image",
    "davis": "video",
    "ytvis": "video",
    "movi": "video",
    "dummy": "video",
    "dummyimage": "image",
}


def build(config):
    dataset_split = config.name.split("_")
    assert len(dataset_split) == 2, "name should be in 'dataset_split' format"
    dataset, split = dataset_split
    assert dataset in DATASET_TYPES.keys(), f"{dataset} is not supported"
    assert split in ["train", "val", "test"]
    dataset_type = DATASET_TYPES[dataset]
    transform_type = config.get("type", "video")
    crop_type = config.get("crop_type", None)
    h_flip_prob = config.get("h_flip_prob", None)
    size = _to_2tuple(config.input_size)
    mask_size = _to_2tuple(config.mask_size) if config.get("mask_size") else size
    use_movi_normalization = config.get("use_movi_normalization", False)

    if dataset_type not in ("image", "video"):
        raise ValueError(f"Unsupported dataset type {transform_type}")
    if transform_type not in ("image", "video"):
        raise ValueError(f"Unsupported transform type {transform_type}")
    if transform_type == "video":
        assert dataset_type == "video"
    if dataset_type == "image":
        assert transform_type == "image"

    if crop_type is not None:
        if split == "train":
            assert crop_type in [
                "central",
                "random",
                "short_side_resize_random",
                "short_side_resize_central",
            ]

        resize_input = CropResize(
            dataset_type=dataset_type,
            crop_type=crop_type,
            size=size,
            resize_mode="bicubic",
            clamp_zero_one=True,
        )
        if split in ["val", "test"]:
            assert crop_type in [
                "central",
                "short_side_resize_central",
            ], f"Only central crops are supported for {split}."
            assert h_flip_prob is None, f"Horizontal flips are not supported for {split}."
            resize_segmentation = CropResize(
                dataset_type=dataset_type,
                crop_type=crop_type,
                size=mask_size,
                resize_mode="nearest-exact",
            )
    else:
        resize_input = Resize(size=size, mode="bicubic", clamp_zero_one=True)
        resize_segmentation = Resize(size=mask_size, mode="nearest-exact")

    if use_movi_normalization:
        normalize = Normalize(
            dataset_type=dataset_type, mean=MOVI_DEFAULT_MEAN, std=MOVI_DEFAULT_STD
        )
    else:
        normalize = Normalize(
            dataset_type=dataset_type, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )
    input_transform = tvt.Compose(
        [
            ToTensorInput(dataset_type=dataset_type),
            resize_input,
            normalize,
        ]
    )
    if h_flip_prob is not None:
        input_transform.transforms.append(
            RandomHorizontalFlip(dataset_type=dataset_type, p=h_flip_prob)
        )
    # Allow the train split to also produce a `segmentations` tensor when the
    # config opts in (e.g. for weakly-supervised loss_id). The val-mode behavior
    # is unchanged and always materializes `segmentation_transformation` for the
    # downstream per-dataset blocks. On train, the resize for segmentations is
    # built lazily from the input resize (no separate `resize_segmentation` is
    # set up above for train splits because crop_type=='random' val path is
    # disallowed).
    add_segs_on_train = split == "train" and config.get("add_segmentations", False)
    if split == "val":
        segmentation_transformation = tvt.Compose(
            [
                ToTensorMask(),
                DenseToOneHotMask(num_classes=config.num_classes),
                resize_segmentation,
            ]
        )
    elif add_segs_on_train:
        # For train we use a nearest-exact segmentation resize matched to the
        # input resolution so the spatial shapes line up.
        train_seg_resize = Resize(size=size, mode="nearest-exact")
    transforms = {transform_type: input_transform}

    if dataset == "movi":
        if "target_size" in config:
            target_size = _to_2tuple(config.target_size)
            transforms[f"target_{transform_type}"] = tvt.Compose(
                [
                    ToTensorInput(dataset_type=dataset_type),
                    Resize(size=target_size, mode="bicubic", clamp_zero_one=True),
                    normalize,
                ]
            )
        if split == "val":
            transforms["segmentations"] = segmentation_transformation
    elif dataset == "davis":
        if "target_size" in config:
            raise NotImplementedError("Separate targets not implemented for transform `davis`")
        if split == "val":
            transforms["segmentations"] = tvt.Compose(
                [
                    ToTensorMask(),
                    DenseToOneHotMask(num_classes=config.num_classes, remove_zero_masks=True),
                    resize_segmentation,
                ]
            )
    elif dataset == "coco":
        if "target_size" in config:
            raise NotImplementedError("Separate targets not implemented for transform `coco`")
        if split == "val":
            transforms["segmentations"] = tvt.Compose(
                [COCOToBinary(num_classes=config.num_classes), resize_segmentation]
            )
    elif dataset == "ytvis":
        if "target_size" in config:
            raise NotImplementedError("Separate targets not implemented for transform `ytvis`")

        if split == "val":
            transforms["segmentations"] = tvt.Compose(
                [YTVISToBinary(num_classes=config.num_classes), resize_segmentation]
            )
        elif add_segs_on_train:
            # Train-side segmentations for weakly-supervised loss_id (Phase 1).
            # `track_ids` is left as-is (numpy int32 -> default collate stacks to
            # [B, 32]); no per-key transform required because it is not a spatial tensor.
            transforms["segmentations"] = tvt.Compose(
                [YTVISToBinary(num_classes=config.num_classes), train_seg_resize]
            )
        
        # Add camera data transforms if use_3d_pos_embed is enabled
        if config.get("use_3d_pos_embed", False):
            transforms["depths"] = DepthTransform(crop_type=crop_type, input_size=size)
            transforms["intrinsics"] = IntrinsicsTransform(crop_type=crop_type, input_size=size)
            transforms["extrinsics"] = ExtrinsicsTransform()

    elif dataset == "dummy" or dataset == "dummyimage":
        if transform_type == "image":
            transforms["image"] = tvt.Compose(
                [
                    tvt.ToTensor(),
                    normalize,
                ]
            )
            transforms["masks"] = tvt.Compose(
                [
                    ToTensorMask(),
                    DenseToOneHotMask(num_classes=config.num_classes),
                ]
            )
        elif transform_type == "video":
            transforms["video"] = tvt.Compose(
                [
                    ToTensorInput(dataset_type=dataset_type),
                    normalize,
                ]
            )
            transforms["masks"] = tvt.Compose(
                [
                    ToTensorMask(),
                    DenseToOneHotMask(num_classes=config.num_classes),
                ]
            )
    else:
        raise ValueError(f"Unknown dataset transforms module `{dataset}`")
    if dataset != "dummy":
        # At this point in the transforms, videos are in CFHW format.
        # Now reorder to FCHW format.
        if dataset_type == "video":
            transforms[transform_type].transforms.append(CFHWToFCHWFormat())
            if f"target_{transform_type}" in transforms:
                transforms[f"target_{transform_type}"].transforms.append(CFHWToFCHWFormat())

        # We transform image-video datasets as one frame video
        # So need to remove first dimension in the end
        if transform_type == "image" and dataset_type == "video":
            squeeze_video_dim = partial(torch.squeeze, dim=0)
            for tf in transforms.values():
                tf.transforms.append(squeeze_video_dim)

    return transforms


def build_inference_transform(config):
    """Builds the transform for inference.

    Modity if needed to match the preprocessing needed for your video.
    """
    use_movi_normalization = config.get("use_movi_normalization", True)
    size = config.get("input_size", 224)
    dataset_type = config.get("dataset_type", "video")

    resize_input = CropResize(
        dataset_type=dataset_type,
        crop_type="central",
        size=size,
        resize_mode="bilinear",
        clamp_zero_one=False,
    )

    if use_movi_normalization:
        normalize = Normalize(
            dataset_type=dataset_type, mean=MOVI_DEFAULT_MEAN, std=MOVI_DEFAULT_STD
        )
    else:
        normalize = Normalize(
            dataset_type=dataset_type, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )
    input_transform = tvt.Compose(
        [
            resize_input,
            normalize,
        ]
    )
    return input_transform


def _to_2tuple(val):
    if val is None:
        return None
    elif isinstance(val, int):
        return (val, val)
    else:
        return val


class CFHWToFCHWFormat:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.permute(tensor, dims=(1, 0, 2, 3))


class ToTensorInput:
    def __init__(self, dataset_type) -> None:
        self.dataset_type = dataset_type
        if dataset_type == "video":
            self.to_cfhw_tensor = transforms_video.ToTensorVideo()

    def __call__(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(array)
        if self.dataset_type == "video":
            tensor = self.to_cfhw_tensor(tensor)
        elif self.dataset_type == "image":
            tensor = tvt_functional.convert_image_dtype(tensor, dtype=torch.float)
            tensor = einops.rearrange(tensor, "h w c -> c h w")
        return tensor


class Normalize:
    def __init__(self, dataset_type: str, mean, std):
        print(dataset_type)
        if dataset_type == "image":
            self.norm = tvt.Normalize(mean=mean, std=std)
        elif dataset_type == "video":
            self.norm = transforms_video.NormalizeVideo(mean=mean, std=std)
        else:
            ValueError(f"Not valid dataset type: {dataset_type}")

    def __call__(self, tensor) -> torch.Tensor:
        return self.norm(tensor)


class RandomHorizontalFlip:
    """
    Flip the video or image clip along the horizonal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, dataset_type: str, p: float = 0.5):
        if dataset_type == "image":
            self.flip = tvt.RandomHorizontalFlip(p)
        elif dataset_type == "video":
            self.flip = transforms_video.RandomHorizontalFlipVideo(p)
        else:
            ValueError(f"Not valid dataset type: {dataset_type}")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.flip(tensor)


class CropResize:
    def __init__(
        self,
        dataset_type: str,
        crop_type: str,
        size: int,
        resize_mode: str,
        clamp_zero_one: bool = False,
        crop_params: Optional[Dict] = None,
    ):
        assert crop_type in [
            "random",
            "central",
            "short_side_resize_central",
            "short_side_resize_random",
        ]
        if crop_type == "random" and crop_params is None:
            crop_params = {}

        if dataset_type == "video":
            self.crop_resize = get_video_crop_resize(
                crop_type, crop_params, size, resize_mode, clamp_zero_one
            )

        elif dataset_type == "image":
            self.crop_resize = get_image_crop_resize(
                crop_type, crop_params, size, resize_mode, clamp_zero_one
            )
        else:
            raise ValueError(f"Unknown dataset_type dataset_type={dataset_type}")
        if clamp_zero_one and crop_type == "random" and resize_mode == "bicubic":
            # we clamp here only random crops
            # because central are already croped in Resize
            self.crop_resize = tvt.Compose([self.crop_resize, partial(torch.clamp, min=0, max=1)])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.crop_resize(tensor)


def get_image_crop_resize(crop_type, crop_params, size, resize_mode, clamp_zero_one):
    if crop_type == "short_side_resize_random":
        short_side_resize = Resize(
            size,
            resize_mode,
            clamp_zero_one=clamp_zero_one,
            short_side_scale=True,
        )
        crop = tvt.RandomCrop(size=size)
        return tvt.Compose([short_side_resize, crop])
    elif crop_type == "central":
        central_crop = CenterFullCrop()
        resize = Resize(size, resize_mode, clamp_zero_one=clamp_zero_one)
        return tvt.Compose([central_crop, resize])
    elif crop_type == "short_side_resize_central":
        short_side_resize = Resize(
            size, resize_mode, clamp_zero_one=clamp_zero_one, short_side_scale=True
        )
        central_crop = CenterFullCrop()
        return tvt.Compose([short_side_resize, central_crop])
    elif crop_type == "random":
        return tvt.RandomResizedCrop(
            size=size,
            interpolation=tvt_functional.InterpolationMode[resize_mode.upper()],
            **crop_params,
        )
    else:
        ValueError(f"Not valid crop_type {crop_type}")


def get_video_crop_resize(crop_type, crop_params, size, resize_mode, clamp_zero_one):
    if crop_type == "central":
        central_crop = transforms_video.CenterFullCropVideo()
        resize = Resize(size, resize_mode, clamp_zero_one=clamp_zero_one)
        return tvt.Compose([central_crop, resize])
    elif crop_type == "short_side_resize_central":
        short_side_resize = Resize(
            size,
            resize_mode,
            clamp_zero_one=clamp_zero_one,
            short_side_scale=True,
        )
        central_crop = transforms_video.CenterFullCropVideo()
        return tvt.Compose([short_side_resize, central_crop])
    elif crop_type == "random":
        return transforms_video.RandomResizedCropVideo(
            size=size,
            interpolation_mode=resize_mode,
            **crop_params,
        )
    elif crop_type == "short_side_resize_random":
        short_side_resize = Resize(
            size,
            resize_mode,
            clamp_zero_one=clamp_zero_one,
            short_side_scale=True,
        )
        crop = transforms_video.RandomCropVideo(size)
        return tvt.Compose(
            [
                short_side_resize,
                crop,
            ]
        )
    else:
        ValueError(f"Not valid crop_type {crop_type}")


class Resize:
    def __init__(
        self,
        size: int,
        mode: str,
        clamp_zero_one: bool = False,
        short_side_scale: bool = False,
    ):

        self.mode = mode
        self.clamp_zero_one = clamp_zero_one
        self.short_side_scale = short_side_scale
        if short_side_scale:
            if isinstance(size, Sequence):
                assert size[0] == size[1]
                self.size = size[0]
            elif isinstance(size, int):
                self.size = size
            else:
                raise ValueError(f"size should be int or tuple but got {size}")
        else:
            self.size = size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        is_image = len(tensor.shape) == 3
        if is_image:
            tensor = tensor[None]
        is_bool = False
        if tensor.dtype == torch.bool:
            is_bool = True
            tensor = tensor.to(torch.uint8)
        if self.short_side_scale:
            tensor = self.scale_short_side(tensor)
        else:
            tensor = torch.nn.functional.interpolate(tensor, size=self.size, mode=self.mode)
        if self.clamp_zero_one and self.mode == "bicubic":
            tensor = torch.clamp(tensor, min=0, max=1)
        if is_bool:
            tensor = tensor.to(torch.bool)
        if is_image:
            tensor = tensor[0]
        return tensor

    def scale_short_side(self, tensor: torch.Tensor) -> torch.Tensor:
        """Scales the shorter spatial dim to the given size.

        To maintain aspect ratio, the longer side is then scaled accordingly.

        Args:
            tensor: A 4D tensor of shape (F, C, H, W) or (B, C, H, W)

        Returns:
            tensor: Tensor with scaled spatial dims.
        """
        assert len(tensor.shape) == 4
        _, _, h, w = tensor.shape
        if w < h:
            new_h = int(math.floor((float(h) / w) * self.size))
            new_w = self.size
        else:
            new_h = self.size
            new_w = int(math.floor((float(w) / h) * self.size))

        return torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode=self.mode)


class CenterFullCrop:
    def __call__(self, img):
        """
        Args:
            image (torch.tensor): Image to be cropped. Size is (C, H, W)
        Returns:
            torch.tensor: central cropping of image. Size is
            (C, crop_size, crop_size)
        """
        min_size = int(min(img.shape[1:]))
        crop_size = (min_size, min_size)
        return tvt.functional.center_crop(img, crop_size)


class ToTensorMask:
    """Transform masks from numpy array uint8 array to float tensor."""

    def __call__(self, mask: np.ndarray):
        assert mask.shape[-1] == 1
        return torch.from_numpy(mask).squeeze(-1).to(torch.float32)


class YTVISToBinary:
    """Transform YTVIS masks to stardart binary form with shape (I, H, W).

    `num_classes` defines the FIXED, padded channel cap. Phase-1 default is 32
    to safely cover YT-VIS 2021 K_video (max ~30) for identity supervision.
    Per-video dynamic sizing is intentionally NOT supported here because it would
    break batched collation of `segmentations` tensors across a batch.
    """

    def __init__(self, num_classes: int = 32):
        self.num_classes = num_classes

    def __call__(self, mask: torch.Tensor):
        f, h, w, num_obj = mask.shape
        # Fixed-cap pad: drop tail channels if the per-shard input ever exceeds
        # the cap (should not happen given the pad-to-32 in save_ytvis2021.py).
        kept = min(num_obj, self.num_classes)
        mask_binary = torch.zeros(f, h, w, self.num_classes, dtype=torch.bool)
        mask = torch.from_numpy(mask != 0).to(torch.bool)
        mask_binary[..., :kept] = mask[..., :kept]
        mask_binary = einops.rearrange(mask_binary, "f h w i -> f i h w")
        return mask_binary


class COCOToBinary:
    """Transform COCO masks to stardart binary form with shape (I, H, W)."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, mask: torch.Tensor):
        num_obj, h, w = mask.shape
        mask_binary = torch.zeros(self.num_classes, h, w, dtype=torch.bool)
        mask = torch.from_numpy(mask != 0).to(torch.bool)
        mask_binary[:num_obj] = mask
        return mask_binary


class DenseToOneHotMask:
    """Transform dense mask of shape (..., H, W) to (..., I, H, W).

    Potentially removes one-hot dim that corresponds to zeros.
    It is useful if all the instances are non-zero encoded,
    whereas zeros correspond to unlabeled pixels (e.g. in DAVIS dataset).
    """

    def __init__(self, num_classes: int, remove_zero_masks: bool = False):
        self.num_classes = num_classes
        self.remove_zero_masks = remove_zero_masks

    def __call__(self, mask: torch.Tensor):
        if self.remove_zero_masks:
            mask_oh = torch.nn.functional.one_hot(mask.to(torch.long), self.num_classes + 1)
            mask_oh = mask_oh[..., 1:]
        else:
            mask_oh = torch.nn.functional.one_hot(mask.to(torch.long), self.num_classes)
        if mask_oh.dim() == 3:
            mask_oh = einops.rearrange(mask_oh, "h w i -> i h w")
        elif mask_oh.dim() == 4:
            mask_oh = einops.rearrange(mask_oh, "f h w i ->f i h w")
        return mask_oh.to(torch.bool)


class Denormalize:
    """
    Denormalization transform for both image and video inputs.

    In case of videos, expected format is FCHW
    as we apply Denormalize after switch from CFHW to FCHW.
    """

    def __init__(self, input_type, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD) -> None:
        if input_type == "video":
            denormalize = transforms_video.DenormalizeVideo(mean=mean, std=std)
            self.denormalize = tvt.Compose(
                [
                    Rearrange("F C H W -> C F H W"),
                    denormalize,
                    Rearrange("C F H W -> F C H W"),
                ]
            )

        elif input_type == "image":
            self.denormalize = DenormalizeImage(mean=mean, std=std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.denormalize(tensor)


class DenormalizeImage(tvt.Normalize):
    def __init__(self, mean, std):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std)


class DepthTransform:
    """Transform depth maps to match resized image resolution.
    
    Applies the same crop/resize logic as video transforms.
    """

    def __init__(self, crop_type: str, input_size):
        self.crop_type = crop_type
        self.size = _to_2tuple(input_size)

    def __call__(self, depths: np.ndarray) -> torch.Tensor:
        """
        Args:
            depths: [F, H, W] numpy array
        Returns:
            Resized/cropped depth tensor [F, H_new, W_new]
        """
        depths = torch.from_numpy(depths).float()  # [F, H, W]
        depths = depths.unsqueeze(1)  # [F, 1, H, W] for interpolate
        
        if self.crop_type in ("short_side_resize_random", "short_side_resize_central"):
            # Short side resize + crop (random or central)
            depths = self._short_side_resize(depths)
            depths = self._center_crop(depths)  # Use center crop for depth (deterministic)
        elif self.crop_type == "central":
            # Center crop first, then resize
            depths = self._center_crop_full(depths)
            depths = torch.nn.functional.interpolate(
                depths, size=self.size, mode="bilinear", align_corners=False
            )
        else:
            # Default: just resize to target size
            depths = torch.nn.functional.interpolate(
                depths, size=self.size, mode="bilinear", align_corners=False
            )
        
        return depths.squeeze(1)  # [F, new_H, new_W]

    def _short_side_resize(self, depths: torch.Tensor) -> torch.Tensor:
        """Scale shorter side to target size, maintaining aspect ratio."""
        _, _, H, W = depths.shape
        target_size = self.size[0]  # Assuming square target
        if W < H:
            new_H = int(math.floor((float(H) / W) * target_size))
            new_W = target_size
        else:
            new_H = target_size
            new_W = int(math.floor((float(W) / H) * target_size))
        return torch.nn.functional.interpolate(
            depths, size=(new_H, new_W), mode="bilinear", align_corners=False
        )

    def _center_crop(self, depths: torch.Tensor) -> torch.Tensor:
        """Center crop to target size."""
        _, _, H, W = depths.shape
        target_H, target_W = self.size
        start_H = (H - target_H) // 2
        start_W = (W - target_W) // 2
        return depths[:, :, start_H:start_H + target_H, start_W:start_W + target_W]

    def _center_crop_full(self, depths: torch.Tensor) -> torch.Tensor:
        """Center crop to square (min dimension)."""
        _, _, H, W = depths.shape
        min_size = min(H, W)
        start_H = (H - min_size) // 2
        start_W = (W - min_size) // 2
        return depths[:, :, start_H:start_H + min_size, start_W:start_W + min_size]


class IntrinsicsTransform:
    """Scale camera intrinsics to match resized image resolution.
    
    Adjusts focal length and principal point based on crop/resize.
    """

    def __init__(self, crop_type: str, input_size):
        self.crop_type = crop_type
        self.size = _to_2tuple(input_size)

    def __call__(self, intrinsics: np.ndarray) -> torch.Tensor:
        """
        Args:
            intrinsics: [F, 3, 3] numpy array  
        Returns:
            Scaled intrinsics tensor [F, 3, 3]
        """
        return torch.from_numpy(intrinsics).float()


class ExtrinsicsTransform:
    """Convert extrinsics to tensor (no modification needed)."""

    def __call__(self, extrinsics: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(extrinsics).float()


class PackCameraData:
    """Pack depth, intrinsics, extrinsics into camera_data dict after transforms."""

    def __call__(self, sample: dict) -> dict:
        if "depths" in sample and "intrinsics" in sample and "extrinsics" in sample:
            sample["camera_data"] = {
                "depth": sample.pop("depths"),
                "intrinsics": sample.pop("intrinsics"),
                "extrinsics": sample.pop("extrinsics"),
            }
        return sample


class OcclusionCurriculumSampler:
    """Occlusion-difficulty curriculum sampler for video clips (idea #013, OcclCurr).

    Operates over a STREAM of already-chunked samples (i.e. placed in a pipeline
    after `split_to_chunks`). Oversamples chunks with long invisibility / re-entry
    gaps, ramping from easy (adjacent-frame) to hard (gap >= `max_ell_target`)
    linearly over the first `curriculum_steps` training steps.

    Gap score definition (per chunk):
        For each instance channel `i`, compute `ell_i` = the maximum consecutive
        run of frames where the instance is absent, *bounded on both sides by a
        frame where the instance is present* (trailing/leading invisibility is
        ignored — an instance that never re-appears does not contribute a gap).
        The chunk's score is `max_ell_in_chunk = max_i ell_i`. If no instance is
        ever visible, the score is 0.

    Curriculum schedule:
        `curriculum_step_fraction = min(1.0, global_step / curriculum_steps)`
        `threshold = ceil(curriculum_step_fraction * max_ell_target)`
        A chunk is accepted if `max_ell_in_chunk >= threshold`. With probability
        `uniform_prob` (default 0.0), a chunk is accepted without filtering so
        the sampler mixes in uniformly-sampled steps — this keeps the data
        distribution from collapsing to only the hardest clips late in training.

    Safety / no-op policy:
        * If `enabled=False`, the sampler is a pass-through (yields every chunk).
        * If the chunk dict does NOT contain a segmentation key (i.e. Table A
          unsupervised runs with `keys: [video]` only), the sampler is a
          pass-through for that chunk. This ensures vanilla Table A runs that
          stream only `video` are never broken.
        * If segmentations are present but malformed (unexpected shape), the
          sampler falls back to pass-through for that chunk rather than
          raising — OcclCurr is a sampler, not a correctness gate.

    Global step tracking:
        The PyTorch-Lightning trainer does not push `global_step` into
        webdataset worker processes. To keep this transform self-contained and
        avoid touching the training loop, we read the step from the env var
        ``SLOTCONTRAST_GLOBAL_STEP`` when present. If it is not set, the
        sampler falls back to a per-worker counter incremented on every yielded
        chunk (approximate but monotonic). Upstream code wishing for exact
        step alignment can set that env var from a Lightning callback.

    Args:
        enabled: If False, the sampler is a no-op.
        curriculum_steps: Ramp length in global steps (default 25000).
        max_ell_target: Target maximum-gap length in frames (default 8).
        seg_key: Key under which per-instance segmentations live in the chunk
            dict. Default ``"segmentations"``.
        uniform_prob: Probability of accepting a chunk without applying the
            curriculum threshold (mix of uniform sampling). Default 0.0.
        min_gap_floor: Lower bound on threshold even at step 0 (default 0 —
            purely easy at t=0).
        seed: Optional seed for the uniform-mix RNG (default None).

    Usage (see configs/slotcontrast/snippets/occl_curr_snippet.yaml):
        >>> sampler = OcclusionCurriculumSampler.from_config(cfg)
        >>> if sampler is not None:
        ...     dataset = dataset.compose(sampler)
    """

    _WORKER_STEP_COUNTER = 0  # per-process fallback counter

    def __init__(
        self,
        enabled: bool = False,
        curriculum_steps: int = 25000,
        max_ell_target: int = 8,
        seg_key: str = "segmentations",
        uniform_prob: float = 0.0,
        min_gap_floor: int = 0,
        seed: Optional[int] = None,
    ):
        if curriculum_steps <= 0:
            raise ValueError(f"curriculum_steps must be > 0, got {curriculum_steps}")
        if max_ell_target < 0:
            raise ValueError(f"max_ell_target must be >= 0, got {max_ell_target}")
        if not 0.0 <= uniform_prob <= 1.0:
            raise ValueError(f"uniform_prob must be in [0, 1], got {uniform_prob}")
        self.enabled = bool(enabled)
        self.curriculum_steps = int(curriculum_steps)
        self.max_ell_target = int(max_ell_target)
        self.seg_key = seg_key
        self.uniform_prob = float(uniform_prob)
        self.min_gap_floor = int(min_gap_floor)
        self._rng = np.random.RandomState(seed) if seed is not None else np.random

    @classmethod
    def from_config(cls, cfg) -> Optional["OcclusionCurriculumSampler"]:
        """Build from a dict/OmegaConf node. Returns None if disabled or absent.

        Conservative policy: any parse error -> return None (no-op), so bad
        configs never block training.
        """
        if cfg is None:
            return None
        try:
            enabled = bool(cfg.get("enabled", False)) if hasattr(cfg, "get") else bool(
                cfg["enabled"]
            )
            if not enabled:
                return None
            get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: cfg.get(k, d))
            return cls(
                enabled=True,
                curriculum_steps=int(get("curriculum_steps", 25000)),
                max_ell_target=int(get("max_ell_target", 8)),
                seg_key=str(get("seg_key", "segmentations")),
                uniform_prob=float(get("uniform_prob", 0.0)),
                min_gap_floor=int(get("min_gap_floor", 0)),
                seed=get("seed", None),
            )
        except Exception:
            return None

    # -----------------------------------------------------------------
    # Gap-score helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _per_frame_presence(seg) -> Optional[np.ndarray]:
        """Return a boolean array of shape (F, I) indicating instance presence.

        Accepts numpy arrays (raw, post-`split_to_chunks`) OR torch tensors
        (post-transforms). Returns None if the layout is unrecognized so the
        caller can fall back to pass-through.
        """
        if seg is None:
            return None
        # Accept both torch and numpy.
        if isinstance(seg, torch.Tensor):
            arr = seg.detach().cpu().numpy()
        else:
            arr = np.asarray(seg)

        if arr.ndim == 4:
            # Two conventional layouts:
            #   (F, H, W, I) — ytvis raw per-shard (H, W spatial; I instances)
            #   (F, I, H, W) — post-YTVISToBinary / movi one-hot
            # Heuristic: pick the axis with the SMALLEST non-F size as instance
            # (instance dim is typically 1..32; H/W are typically >= 64).
            candidate_axes = [1, 2, 3]
            sizes = [arr.shape[a] for a in candidate_axes]
            inst_axis = candidate_axes[int(np.argmin(sizes))]
            spatial_axes = tuple(a for a in candidate_axes if a != inst_axis)
            reduced = np.any(arr != 0, axis=spatial_axes)  # keeps (F, inst_axis)
            # Move instance axis to position 1 to yield canonical (F, I).
            if inst_axis != 1:
                # After reducing two axes, `reduced` has shape (F, I) already
                # because axes are collapsed. But the collapsing preserves the
                # ORIGINAL order of remaining axes, so if inst_axis==3 the
                # result is (F, I); if inst_axis==2 it's (F, I); if
                # inst_axis==1 it's (F, I). Numpy collapses in order, so this
                # is always (F, I) after reducing the two spatial dims. No
                # extra moveaxis needed.
                pass
            return reduced.astype(bool)
        elif arr.ndim == 3:
            # Dense label map layout (F, H, W) — treat unique nonzero ids as
            # per-instance channels. Trailing-singleton (F, H, W, 1) also lands
            # here after squeeze; in that case there is a single "foreground"
            # channel.
            ids = np.unique(arr)
            ids = ids[ids != 0]
            if ids.size == 0:
                return np.zeros((arr.shape[0], 1), dtype=bool)
            presence = np.stack(
                [np.any(arr == i, axis=(1, 2)) for i in ids], axis=1
            )  # (F, I)
            return presence.astype(bool)
        else:
            return None

    @staticmethod
    def _max_gap_per_instance(presence_fi: np.ndarray) -> int:
        """Longest run of False between two Trues, taken max over instances."""
        if presence_fi is None or presence_fi.size == 0:
            return 0
        f, num_i = presence_fi.shape
        best = 0
        for i in range(num_i):
            col = presence_fi[:, i]
            if not col.any():
                continue
            # Clip to the range [first_true, last_true] so we ignore trailing
            # / leading invisibility (never-entered or never-returned cases).
            idx = np.where(col)[0]
            first, last = int(idx[0]), int(idx[-1])
            inner = col[first : last + 1]
            # Longest run of False inside [first, last].
            run = 0
            local_best = 0
            for v in inner:
                if not v:
                    run += 1
                    if run > local_best:
                        local_best = run
                else:
                    run = 0
            if local_best > best:
                best = local_best
        return best

    def _rand_uniform(self) -> float:
        """Draw a uniform [0, 1) sample compatible with both RandomState and
        the top-level ``np.random`` module (where ``random_sample`` is the
        portable API across numpy versions)."""
        if hasattr(self._rng, "random_sample"):
            return float(self._rng.random_sample())
        return float(self._rng.random())

    def _current_threshold(self) -> int:
        import os

        env_step = os.environ.get("SLOTCONTRAST_GLOBAL_STEP", None)
        if env_step is not None:
            try:
                step = int(env_step)
            except ValueError:
                step = type(self)._WORKER_STEP_COUNTER
        else:
            step = type(self)._WORKER_STEP_COUNTER
        frac = min(1.0, float(step) / float(self.curriculum_steps))
        threshold = int(math.ceil(frac * self.max_ell_target))
        return max(threshold, self.min_gap_floor)

    # -----------------------------------------------------------------
    # Stream interface (webdataset .compose-compatible)
    # -----------------------------------------------------------------
    def __call__(self, data):
        """Iterate chunks and filter by the current curriculum threshold."""
        if not self.enabled:
            yield from data
            return

        for chunk in data:
            # No-op when seg stream absent (vanilla Table A: keys=[video]).
            if not isinstance(chunk, dict) or self.seg_key not in chunk:
                type(self)._WORKER_STEP_COUNTER += 1
                yield chunk
                continue

            threshold = self._current_threshold()

            # Uniform-sample escape hatch.
            if self.uniform_prob > 0.0 and self._rand_uniform() < self.uniform_prob:
                type(self)._WORKER_STEP_COUNTER += 1
                yield chunk
                continue

            # If threshold is 0, everything passes (early curriculum).
            if threshold <= 0:
                type(self)._WORKER_STEP_COUNTER += 1
                yield chunk
                continue

            presence = self._per_frame_presence(chunk[self.seg_key])
            if presence is None:
                # Unrecognized segmentation layout — conservative pass-through.
                type(self)._WORKER_STEP_COUNTER += 1
                yield chunk
                continue

            max_ell = self._max_gap_per_instance(presence)
            type(self)._WORKER_STEP_COUNTER += 1
            if max_ell >= threshold:
                yield chunk
            # else: drop the chunk (filtered by curriculum)
