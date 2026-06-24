import math
from typing import Dict, Optional, Sequence, Tuple

import einops
import numpy as np
import torch
import torchmetrics

from slotcontrast.utils import make_build_fn


@make_build_fn(__name__, "metric")
def build(config, name: str):
    pass  # No special module building needed


class Metric(torchmetrics.Metric):
    def __init__(self, input_mapping: Dict[str, str], **kwargs) -> None:
        super().__init__(**kwargs)
        # Mapping from parameter in _update to name in inputs dict
        self.input_mapping = input_mapping

    def update(self, *args, **kwargs):
        inputs = {}
        for mapped_key, input_key in self.input_mapping.items():
            if input_key is None:
                continue
            if input_key not in kwargs:
                raise ValueError(
                    f"Key {input_key} not found in inputs to metric. "
                    f"Available inputs are: {list(kwargs)}"
                )
            inputs[mapped_key] = kwargs[input_key]

        return self._update(*args, **inputs)

    def _update(self, *args, **kwargs):
        raise NotImplementedError("Implement in subclasses")


class ImageMaskMetricMixin:
    """Mixin class for mask-based per-image metrics.

    Handles shape checking and rearranging of inputs.

    Args:
        video_input: If true, assumes additional frame dimension as input.
            Each frame is treated as an independent image for ARI computation.
        flatten_spatially: If true, flatten spatial dimensions into a single dimension.
        move_classes_last: If true, move classes to the last dimension.
    """

    def __init__(
        self,
        video_input: bool,
        *args,
        flatten_spatially: bool = True,
        move_classes_last: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.video_input = video_input
        self.rearrange_pattern = self._get_rearrange_pattern(
            video_input, flatten_spatially, move_classes_last
        )

    @staticmethod
    def _get_rearrange_pattern(video_input, flatten_spatially, move_classes_last) -> str:
        if video_input:
            # For video input, the temporal dimension is folded into the batch dimension, i.e.
            # all frames are treated independently from the downstream metrics.
            if flatten_spatially:
                if move_classes_last:
                    pattern = "b t c h w -> (b t) (h w) c"
                else:
                    pattern = "b t c h w -> (b t) c (h w)"
            else:
                if move_classes_last:
                    pattern = "b t c h w -> (b t) h w c"
                else:
                    pattern = "b t c h w -> (b t) c h w"
        else:
            if flatten_spatially:
                if move_classes_last:
                    pattern = "b c h w -> b (h w) c"
                else:
                    pattern = "b c h w -> b c (h w)"
            else:
                if move_classes_last:
                    pattern = "b c h w -> b h w c"
                else:
                    pattern = "b c h w -> b c h w"

        return pattern

    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        """Update metric.

        Args:
            true_mask: Binary true masks of shape (batch [, n_frames], n_true_classes, height,
                width).
            pred_mask: One-hot predicted masks of shape (batch [, n_frames], n_pred_classes, height,
                width).
        """
        if self.video_input:
            _check_shape(
                true_mask,
                (None, None, None, None, None),
                "true_mask [bs, n_frames, n_true_classes, h, w]",
            )
            b, t, _, h, w = true_mask.shape
            _check_shape(
                pred_mask,
                (b, t, None, h, w),
                "pred_mask [bs, n_frames, n_pred_classes, h, w]",
            )
        else:
            _check_shape(true_mask, (None, None, None, None), "true_mask [bs, n_true_classes, h, w]")
            b, _, h, w = true_mask.shape
            _check_shape(pred_mask, (b, None, h, w), "pred_mask [bs, n_pred_classes, h, w]")

        true_mask = einops.rearrange(true_mask, self.rearrange_pattern)
        pred_mask = einops.rearrange(pred_mask, self.rearrange_pattern)

        return super()._update(true_mask, pred_mask)


class VideoMaskMetricMixin:
    """Mixin class for mask-based per-video metrics.

    Handles shape checking and rearranging of inputs.

    Args:
        flatten_temporally: If true, flatten temporal dimensions into the spatial dimensions.
            In this case, frames are vertically concatenated.
        flatten_spatially: If true, flatten spatial dimensions into a single dimension.
        move_classes_last: If true, move classes to the last dimension.
    """

    def __init__(
        self,
        *args,
        flatten_temporally: bool = True,
        flatten_spatially: bool = True,
        move_classes_last: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rearrange_pattern = self._get_rearrange_pattern(
            flatten_temporally, flatten_spatially, move_classes_last
        )

    @staticmethod
    def _get_rearrange_pattern(
        flatten_temporally: bool, flatten_spatially: bool, move_classes_last: bool
    ) -> str:
        if flatten_temporally:
            if flatten_spatially:
                if move_classes_last:
                    pattern = "b t c h w -> b (t h w) c"
                else:
                    pattern = "b t c h w -> b c (t h w)"
            else:
                # Temporal dimension is folded into the height dimension, i.e. all frames are
                # vertically concatenated.
                if move_classes_last:
                    pattern = "b t c h w -> b (t h) w c"
                else:
                    pattern = "b t c h w -> b c (t h) w"
        else:
            if flatten_spatially:
                if move_classes_last:
                    pattern = "b t c h w -> b t (h w) c"
                else:
                    pattern = "b t c h w -> b t c (h w)"
            else:
                if move_classes_last:
                    pattern = "b t c h w -> b t h w c"
                else:
                    pattern = "b t c h w -> b t c h w"

        return pattern

    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        """Update metric.

        Args:
            true_mask: Binary true masks of shape (batch, n_frames, n_true_classes, height, width).
            pred_mask: One-hot predicted masks of shape (batch, n_frames, n_pred_classes, height,
                width).
        """
        _check_shape(
            true_mask,
            (None, None, None, None, None),
            "true_mask [bs, n_frames, n_true_classes, h, w]",
        )
        b, t, _, h, w = true_mask.shape
        _check_shape(
            pred_mask,
            (b, t, None, h, w),
            "pred_mask [bs, n_frames, n_pred_classes, h, w]",
        )
        true_mask = einops.rearrange(true_mask, self.rearrange_pattern)
        pred_mask = einops.rearrange(pred_mask, self.rearrange_pattern)

        return super()._update(true_mask, pred_mask)


class AdjustedRandIndex(Metric):
    """Abstract ARI metric."""

    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(input_mapping={"pred_mask": pred_key, "true_mask": true_key})
        self.ignore_background = ignore_background
        self.ignore_overlaps = ignore_overlaps
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        """Update metric.

        Args:
            true_mask: Binary true masks of shape (batch, n_points, n_true_classes)
            pred_mask: One-hot predicted masks of shape (batch, n_points, n_pred_classes)
        """
        assert true_mask.ndim == 3
        assert pred_mask.ndim == 3
        if torch.any((true_mask != 0.0) & (true_mask != 1.0)):
            raise ValueError("`true_mask` is not binary")
        if torch.any((pred_mask != 0.0) & (pred_mask != 1.0)):
            raise ValueError("`pred_mask` is not binary")
        if torch.any(pred_mask.sum(dim=-1) != 1.0):
            raise ValueError("`pred_mask` is not one-hot")

        n_true_classes_per_point = true_mask.sum(dim=-1)
        if not self.ignore_overlaps and torch.any(n_true_classes_per_point > 1.0):
            raise ValueError("There are overlaps in `true_mask`.")

        if self.ignore_overlaps:
            overlaps = n_true_classes_per_point > 1.0
            true_mask = true_mask.clone()
            true_mask[overlaps] = 0.0  # ARI ignores pixels where all ground truth clusters are zero

        if self.ignore_background:
            true_mask = true_mask[..., 1:]  # Remove the background mask

        values = adjusted_rand_index(true_mask, pred_mask)

        # Special case: skip samples without any ground truth mask
        non_empty = n_true_classes_per_point.sum(dim=-1) > 0
        values = values[non_empty]

        self.values += values.sum()
        self.total += len(values)

    def compute(self):
        return self.values / self.total


class ImageARI(ImageMaskMetricMixin, AdjustedRandIndex):
    """ARI metric for images.

    Inputs to metric:
        true_mask: Binary true masks of shape (batch [, n_frames], n_true_classes, height,
            width).
        pred_mask: One-hot predicted masks of shape (batch [, n_frames], n_pred_classes, height,
            width).

    Args:
        video_input: If true, assumes additional frame dimension as input.
            Each frame is treated as an independent image for metric computation.
        ignore_background: If true, assume first dimension of true masks is background to ignore.
        ignore_overlaps: If true, ignore pixels from overlapping instances.
    """

    def __init__(
        self,
        video_input: bool = False,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(video_input, ignore_background, ignore_overlaps, pred_key, true_key)


class VideoARI(VideoMaskMetricMixin, AdjustedRandIndex):
    """ARI metric for videos.

    Inputs to metric:
        true_mask: Binary true masks of shape (batch, n_frames, n_true_classes, height, width).
        pred_mask: One-hot predicted masks of shape (batch, n_frames, n_pred_classes, height,
            width).

    Args:
        ignore_background: If true, assume first dimension of true masks is background to ignore.
        ignore_overlaps: If true, ignore pixels from overlapping instances.
    """

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(ignore_background, ignore_overlaps, pred_key, true_key)


def adjusted_rand_index(
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
) -> torch.Tensor:
    """Computes the adjusted Rand index (ARI), a clustering similarity score.

    Adapted to Pytorch from SAVi Jax implementation:
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

    Args:
        true_mask: A binary tensor of shape (batch_size, n_points, n_true_clusters). The true cluster
            assignment encoded as one-hot with missing values allowed.
        pred_mask: A binary tensor of shape (batch_size, n_points, n_pred_clusters). The predicted
            cluster assignment encoded as one-hot.

    Returns:
        ARI scores as a tensor of shape (batch_size,).
    """
    N = torch.einsum("bpc, bpk -> bck", true_mask.to(torch.float64), pred_mask.to(torch.float64))
    A = torch.sum(N, axis=-1)  # row-sum  (batch_size, c)
    B = torch.sum(N, axis=-2)  # col-sum  (batch_size, k)
    num_points = torch.sum(A, axis=1)

    rindex = torch.sum(N * (N - 1), axis=[1, 2])
    aindex = torch.sum(A * (A - 1), axis=1)
    bindex = torch.sum(B * (B - 1), axis=1)
    expected_rindex = aindex * bindex / torch.clip(num_points * (num_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both label_pred and label_true assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == num_points * (num_points-1))
    # 2. If both label_pred and label_true assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator != 0.0, ari, 1.0)


class IntersectionOverUnion(Metric):
    """Abstract IoU metric."""

    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        matching: str = "none",
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(input_mapping={"pred_mask": pred_key, "true_mask": true_key})
        self.ignore_background = ignore_background
        self.ignore_overlaps = ignore_overlaps
        self.matching = matching
        if matching not in ("none", "overlap", "hungarian"):
            raise ValueError("`matching` needs to be 'none' or 'overlap' or 'hungarian'")
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        """Update metric.

        Args:
            true_mask: Binary true masks of shape (batch, n_points, n_true_classes)
            pred_mask: One-hot predicted masks of shape (batch, n_points, n_pred_classes)
        """
        assert true_mask.ndim == 3
        assert pred_mask.ndim == 3
        if torch.any((true_mask != 0.0) & (true_mask != 1.0)):
            raise ValueError("`true_mask` is not binary")
        if torch.any((pred_mask != 0.0) & (pred_mask != 1.0)):
            raise ValueError("`pred_mask` is not binary")
        if torch.any(pred_mask.sum(dim=-1) != 1.0):
            raise ValueError("`pred_mask` is not one-hot")

        n_true_classes_per_point = true_mask.sum(dim=-1)
        if not self.ignore_overlaps and torch.any(n_true_classes_per_point > 1.0):
            raise ValueError("There are overlaps in `true_mask`.")
        if self.ignore_overlaps:
            overlaps = n_true_classes_per_point > 1.0
            true_mask = true_mask.clone()
            true_mask[overlaps] = 0.0
            pred_mask = pred_mask.clone()
            pred_mask[overlaps] = 0.0

        if self.ignore_background:
            true_mask = true_mask[..., 1:]  # Remove the background mask

        values = intersection_over_union_with_matching(
            true_mask, pred_mask, self.matching, empty_value=0.0
        )
        active_true_classes = true_mask.sum(dim=1) > 0

        # Compute mean IoU, ignoring empty true classes. This assumes that true-pred class pairs
        # with union==0 have been assigned zero IoU.
        n_true_classes = active_true_classes.sum(dim=-1)
        mean_iou = values.sum(dim=-1) / n_true_classes

        # Special case: skip samples without any ground truth mask
        non_empty = n_true_classes > 0
        mean_iou = mean_iou[non_empty]

        self.values += mean_iou.sum()
        self.total += len(mean_iou)

    def compute(self):
        return self.values / self.total


class ImageIoU(ImageMaskMetricMixin, IntersectionOverUnion):
    """IoU metric for images.

    Inputs to metric:
        true_mask: Binary true masks of shape (batch [, n_frames], n_true_classes, height,
            width).
        pred_mask: One-hot predicted masks of shape (batch [, n_frames], n_pred_classes, height,
            width).

    Args:
        video_input: If true, assumes additional frame dimension as input.
            Each frame is treated as an independent image for metric computation.
        ignore_background: If true, assume first dimension of true masks is background to ignore.
        ignore_overlaps: If true, ignore pixels from overlapping instances.
        matching: How to match true classes to predicted classes. For "none", assume classes are
            ordered, i.e. the true class at index i corresponds to the predicted class at index i.
            For "overlap", match the predicted class with the highest IoU to each true class.
    """

    def __init__(
        self,
        video_input: bool = False,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        matching: str = "none",
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(
            video_input, ignore_background, ignore_overlaps, matching, pred_key, true_key
        )


class VideoIoU(VideoMaskMetricMixin, IntersectionOverUnion):
    """IoU metric for videos.

    Inputs to metric:
        true_mask: Binary true masks of shape (batch, n_frames, n_true_classes, height, width).
        pred_mask: One-hot predicted masks of shape (batch, n_frames, n_pred_classes, height,
            width).

    Args:
        ignore_background: If true, assume first dimension of true masks is background to ignore.
        ignore_overlaps: If true, ignore pixels from overlapping instances.
        matching: How to match true classes to predicted classes. For "none", assume classes are
            ordered, i.e. the true class at index i corresponds to the predicted class at index i.
            For "overlap", match the predicted class with the highest IoU to each true class.
    """

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        matching: str = "none",
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(ignore_background, ignore_overlaps, matching, pred_key, true_key)


def intersection_over_union_with_matching(
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    matching: str = "none",
    empty_value: float = 0.0,
) -> torch.Tensor:
    """Computes mask intersection-over-union, matching predicted masks to ground truth masks.

    Args:
        true_mask: A binary tensor of shape (batch_size, n_points, n_true_classes). The true class
            mask encoded as one-hot with missing values allowed.
        pred_mask: A binary tensor of shape (batch_size, n_points, n_pred_classes). The predicted
            class mask encoded as one-hot with missing values allowed.
        matching: How to match true classes to predicted classes. For "none", assume classes are
            ordered, i.e. the true class at index i corresponds to the predicted class at index i.
            For "overlap", match the predicted class with the highest IoU to each true class.
        empty_value: Value to assume for the case when a class does not occur in the ground truth,
            and was also not predicted.

    Returns:
        IoU scores as a tensor of shape (batch_size, n_true_classes).
    """
    assert matching in ("none", "overlap", "hungarian")

    pairwise_ious = intersection_over_union(true_mask, pred_mask, empty_value)

    if matching == "none":
        if pairwise_ious.shape[1] != pairwise_ious.shape[2]:
            raise ValueError(
                "For matching 'none', n_true_classes needs to equal n_pred_classes, but is "
                f"{pairwise_ious.shape[1]} vs {pairwise_ious.shape[2]}"
            )
        ious = torch.diagonal(pairwise_ious, dim1=-2, dim2=-1)
    elif matching == "overlap":
        ious = torch.max(pairwise_ious, dim=2).values
    elif matching == "hungarian":
        all_true_idxs, all_pred_idxs = hungarian_matching(pairwise_ious, maximize=True)
        ious = torch.zeros(
            true_mask.shape[0], true_mask.shape[2], dtype=torch.float64, device=pairwise_ious.device
        )
        for idx, (true_idxs, pred_idxs) in enumerate(zip(all_true_idxs, all_pred_idxs)):
            ious[idx, true_idxs] = pairwise_ious[idx, true_idxs, pred_idxs]
    else:
        raise ValueError(f"Unknown matching for IoU `{matching}`")

    assert ious.shape == (true_mask.shape[0], true_mask.shape[2])
    return ious


def intersection_over_union(
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    empty_value: float = 0.0,
):
    """Compute pairwise intersection-over-union between predicted and ground truth masks.

    Args:
        true_mask: A binary tensor of shape (batch_size, n_points, n_true_classes). The true class
            mask encoded as one-hot with missing values allowed.
        pred_mask: A binary tensor of shape (batch_size, n_points, n_pred_classes). The predicted
            class mask encoded as one-hot with missing values allowed.
        empty_value: Value to assume for the case when a class does not occur in the ground truth,
            and was also not predicted.

    Returns:
        Pairwise IoU scores as a tensor of shape (batch_size, n_true_classes, n_pred_classes)."""
    intersection, fp, fn = confusion_matrix(true_mask, pred_mask)  # all B x C x K
    union = intersection + fp + fn
    ious = intersection / union

    # Deal with NaN from divide-by-zero (class does not occur and was not predicted)
    ious[union == 0] = empty_value

    return ious


def confusion_matrix(true_mask: torch.Tensor, pred_mask: torch.Tensor):
    """Computes confusion matrix between two sets of masks.

    Args:
        true_mask: A binary tensor of shape (batch_size, n_points, n_true_classes). The true class
            mask encoded as one-hot with missing values allowed.
        pred_mask: A binary tensor of shape (batch_size, n_points, n_pred_classes). The predicted
            class mask encoded as one-hot with missing values allowed.

    Returns:
        Tuple containing the pairwise true positives (intersection), false positives and false
        negatives, all of shape (batch_size, n_true_classes, n_pred_classes).
    """
    true_mask = true_mask.to(torch.float64)
    pred_mask = pred_mask.to(torch.float64)

    true_positives = torch.einsum("bpc, bpk -> bck", true_mask, pred_mask)  # B x C x K
    n_true_points = true_mask.sum(1)  # B x C
    n_pred_points = pred_mask.sum(1)  # B x K

    false_positives = n_pred_points.unsqueeze(1) - true_positives  # B x C x K
    false_negatives = n_true_points.unsqueeze(2) - true_positives  # B x C x K

    return true_positives, false_positives, false_negatives


class JandFMetric(Metric):
    """Abstract Jaccard and F-score metric used in video object discovery.

    See also official implementation from DAVIS challenge:
    https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/metrics.py
    """

    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
        metric_for_matching: str = "j_and_f",
    ):
        super().__init__(input_mapping={"pred_mask": pred_key, "true_mask": true_key})
        self.ignore_background = ignore_background
        self.ignore_overlaps = ignore_overlaps
        if metric_for_matching not in ("j_and_f", "jaccard", "f_measure"):
            raise ValueError(
                "Matching should be one of 'j_and_f', 'jaccard', 'f_measure', but is "
                f"{metric_for_matching}"
            )
        self.metric_for_matching = metric_for_matching
        self.add_state(
            "j_and_f", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "jaccard", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "f_measure", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def _aggregate_from_pairwise_values(pairwise_values, true_idxs, pred_idxs, n_true_classes):
        """

        Args:
            values: tensor of shape (batch, n_true_classes, n_pred_classes)
            true_idxs: tensor of shape (batch, n_true_classes)
            pred_idxs: tensor of shape (batch, n_pred_classes)
            n_true_classes: tensor of shape (batch,)

        Returns:
            Tensor of shape (batch_filtered,) where batch_filtered is batch size minus the number
            of true masks without any active mask, containing the mean metric value per sample.
        """
        values = torch.zeros(
            pairwise_values.shape[:2], dtype=torch.float64, device=pairwise_values.device
        )
        for idx, (t, p) in enumerate(zip(true_idxs, pred_idxs)):
            values[idx, t] = pairwise_values[idx, t, p]

        # Compute mean values, ignoring empty true masks. This assumes that empty masks have been
        # assigned a value of zero.
        values = values.sum(dim=-1) / n_true_classes

        # Special case: skip samples without any ground truth mask
        non_empty = n_true_classes > 0
        values = values[non_empty]

        return values

    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        """Update metric.

        Args:
            true_mask: Binary true masks of shape (batch, [n_frames,] n_true_classes, height, width)
            pred_mask: One-hot predicted masks of shape (batch, [n_frames,] n_pred_classes, height,
                width)
        """
        assert true_mask.ndim == pred_mask.ndim == 4 or true_mask.ndim == pred_mask.ndim == 5
        batch_size = len(true_mask)
        if true_mask.ndim == 4:
            # Insert dummy video dimension if we are passed images
            true_mask = true_mask.unsqueeze(1)
            pred_mask = pred_mask.unsqueeze(1)

        if torch.any((true_mask != 0.0) & (true_mask != 1.0)):
            raise ValueError("`true_mask` is not binary")
        if torch.any((pred_mask != 0.0) & (pred_mask != 1.0)):
            raise ValueError("`pred_mask` is not binary")
        if torch.any(pred_mask.sum(dim=2) != 1.0):
            raise ValueError("`pred_mask` is not one-hot")

        n_true_classes_per_point = true_mask.sum(dim=2)
        if not self.ignore_overlaps and torch.any(n_true_classes_per_point > 1.0):
            raise ValueError("There are overlaps in `true_mask`.")
        if self.ignore_background and torch.any(n_true_classes_per_point != 1.0):
            raise ValueError("`true_mask` is not one-hot")
        if self.ignore_overlaps:
            overlaps = n_true_classes_per_point > 1.0
            true_mask = true_mask.clone()
            m = einops.repeat(overlaps, "b t h w -> b t c h w", c=true_mask.shape[2])
            true_mask[m] = 0.0
            pred_mask = pred_mask.clone()
            m = einops.repeat(overlaps, "b t h w -> b t k h w", k=pred_mask.shape[2])
            pred_mask[m] = 0.0

        if self.ignore_background:
            true_mask = true_mask[:, :, 1:]  # Remove the background mask

        # Jaccard is computed frame-by-frame in the original J & F implementation.
        all_jaccard = intersection_over_union(
            einops.rearrange(true_mask, "b t c h w -> (b t) (h w) c"),
            einops.rearrange(pred_mask, "b t k h w -> (b t) (h w) k"),
            empty_value=0.0,
        )  # B x true_classes x pred_classes
        all_jaccard = einops.rearrange(all_jaccard, "(b t) c k -> b t c k", b=batch_size)
        all_jaccard = all_jaccard.mean(1)

        # Boundary f-measure is computed frame-by-frame
        all_f_measure, _, _ = boundary_f_measure(
            einops.rearrange(true_mask, "b t c h w -> (b t) c h w"),
            einops.rearrange(pred_mask, "b t k h w -> (b t) k h w"),
        )  # (B * frames) x true_classes x pred_classes
        all_f_measure = einops.rearrange(all_f_measure, "(b t) c k -> b t c k", b=batch_size)
        all_f_measure = all_f_measure.mean(1)

        all_j_and_f = (all_jaccard + all_f_measure) / 2

        if self.metric_for_matching == "j_and_f":
            true_idxs, pred_idxs = hungarian_matching(all_j_and_f, maximize=True)
        elif self.metric_for_matching == "jaccard":
            true_idxs, pred_idxs = hungarian_matching(all_jaccard, maximize=True)
        elif self.metric_for_matching == "f_measure":
            true_idxs, pred_idxs = hungarian_matching(all_f_measure, maximize=True)
        else:
            raise ValueError(f"Unknown matching {self.metric_for_matching}")

        active_true_classes = true_mask.sum(dim=(1, -2, -1)) > 0  # B x true_classes
        n_true_classes = active_true_classes.sum(dim=-1)

        j_and_f = self._aggregate_from_pairwise_values(
            all_j_and_f, true_idxs, pred_idxs, n_true_classes
        )
        jaccard = self._aggregate_from_pairwise_values(
            all_jaccard, true_idxs, pred_idxs, n_true_classes
        )
        f_measure = self._aggregate_from_pairwise_values(
            all_f_measure, true_idxs, pred_idxs, n_true_classes
        )

        self.j_and_f += j_and_f.sum()
        self.jaccard += jaccard.sum()
        self.f_measure += f_measure.sum()
        self.total += len(j_and_f)

    def compute(self):
        return {
            "j_and_f": self.j_and_f / self.total,
            "jaccard": self.jaccard / self.total,
            "boundary_f_measure": self.f_measure / self.total,
        }


class ImageJandF(ImageMaskMetricMixin, JandFMetric):
    """J&F metric for images.

    Inputs to metric:
        true_mask: Binary true masks of shape (batch [, n_frames], n_true_classes, height,
            width).
        pred_mask: One-hot predicted masks of shape (batch [, n_frames], n_pred_classes, height,
            width).

    Args:
        video_input: If true, assumes additional frame dimension as input.
            Each frame is treated as an independent image for metric computation.
        ignore_background: If true, assume first dimension of true masks is background to ignore.
        ignore_overlaps: If true, ignore pixels from overlapping instances.
    """

    def __init__(
        self,
        video_input: bool = False,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
        metric_for_matching: str = "j_and_f",
    ):
        super().__init__(
            video_input,
            ignore_background,
            ignore_overlaps,
            pred_key,
            true_key,
            flatten_spatially=False,
            move_classes_last=False,
            metric_for_matching=metric_for_matching,
        )


class VideoJandF(VideoMaskMetricMixin, JandFMetric):
    """J&F metric for videos.

    Inputs to metric:
        true_mask: Binary true masks of shape (batch, n_frames, n_true_classes, height, width).
        pred_mask: One-hot predicted masks of shape (batch, n_frames, n_pred_classes, height,
            width).

    Args:
        ignore_background: If true, assume first dimension of true masks is background to ignore.
        ignore_overlaps: If true, ignore pixels from overlapping instances.
    """

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
        metric_for_matching: str = "j_and_f",
    ):
        super().__init__(
            ignore_background,
            ignore_overlaps,
            pred_key,
            true_key,
            flatten_temporally=False,
            flatten_spatially=False,
            move_classes_last=False,
            metric_for_matching=metric_for_matching,
        )


def boundary_f_measure(true_mask: torch.Tensor, pred_mask: torch.Tensor):
    """Computes pairwise F-measure on boundaries of masks.

    Args:
        true_mask: A binary tensor of shape (batch_size, n_true_classes, height, width). The true
            masks encoded as one-hot with missing values allowed.
        pred_mask: A binary tensor of shape (batch_size, n_pred_classes, height, width). The
            predicted masks encoded as one-hot with missing values allowed.

    Returns:
        Tuple containing the pairwise f-measure, precision and recall, all of shape (batch_size,
        n_true_classes, n_pred_classes).
    """
    true_boundaries = einops.rearrange(masks_to_boundaries(true_mask), "b c h w -> b (h w) c")
    pred_boundaries = einops.rearrange(masks_to_boundaries(pred_mask), "b c h w -> b (h w) c")
    return f_measure(true_boundaries, pred_boundaries)


def f_measure(true_mask: torch.Tensor, pred_mask: torch.Tensor, empty_value: float = 0.0):
    """Compute pairwise F-measure between predicted and ground truth masks.

    Args:
        true_mask: A binary tensor of shape (batch_size, n_points, n_true_classes). The true class
            mask encoded as one-hot with missing values allowed.
        pred_mask: A binary tensor of shape (batch_size, n_points, n_pred_classes). The predicted
            class mask encoded as one-hot with missing values allowed.
        empty_value: Value to assume for the case when a class does not occur in the ground truth,
            or in the predicted classes.

    Returns:
        Tuple containing the pairwise f-measure, precision and recall, all of shape (batch_size,
        n_true_classes, n_pred_classes).
    """
    tp, fp, fn = confusion_matrix(true_mask, pred_mask)  # all B x C x K
    n_pred_points = tp + fp
    n_true_points = tp + fn

    precision = tp / n_pred_points
    recall = tp / n_true_points

    # Guard against NaNs
    precision[n_pred_points == 0] = empty_value
    recall[n_true_points == 0] = empty_value

    f_measure = 2 * precision * recall / (precision + recall)
    f_measure[precision + recall == 0] = 0

    return f_measure, precision, recall


def masks_to_boundaries(masks: torch.Tensor, dilation_ratio: float = 0.02) -> torch.Tensor:
    """Compute the boundaries around the provided masks using morphological operations.

    Returns a tensor of the same shape as the input masks containing the boundaries of each mask.

    This implementations is adapted from a non-merged PR to torchvision, see
    https://github.com/pytorch/vision/pull/7704/.

    Args:
        masks: masks to transform of shape ([batch_size,], n_masks, height, width).
        dilation_ratio: ratio used for the dilation operation.

    Returns:
        Tensor of shape ([batch_size,], n_masks, height, width) with boundaries.
    """
    # If no masks are provided, return an empty tensor
    if masks.numel() == 0:
        return torch.zeros_like(masks)

    orig_dtype = masks.dtype
    if masks.ndim == 4:
        # Flatten batch dim into mask dim
        batch_size, n_masks = masks.shape[:2]
        masks = masks.flatten(0, 1)
    else:
        batch_size = None
    _, h, w = masks.shape
    img_diag = math.sqrt(h**2 + w**2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))
    selem_size = dilation * 2 + 1
    selem = torch.ones((1, 1, selem_size, selem_size), device=masks.device)

    # Compute the boundaries for each mask
    masks = masks.float().unsqueeze(1)
    # The original erosion operation is emulated in two steps:
    #   1) Convolution counting the number of black pixels around each pixel
    #   2) Zero out pixel with less than the full number of black pixels
    # Step 2) effectively performs the minimum operation in erosion.
    eroded_masks = torch.nn.functional.conv2d(masks, selem, padding=dilation)
    eroded_masks = (eroded_masks == selem.sum()).byte()

    contours = masks.byte() - eroded_masks
    contours = contours.squeeze(1)

    if batch_size is not None:
        contours = contours.unflatten(0, (batch_size, n_masks))

    return contours.to(orig_dtype)


def hungarian_matching(
    cost_matrix: torch.Tensor, maximize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bipartite graph matching using Hungarian method.

    Naively computed over batches with a loop.

    Args:
        cost_matrix: Tensor of shape (batch_size, n_rows, n_cols) containing costs.

    Returns:
        Tuple of tensors of shape (batch_size, min_rows_cols, min_rows_cols) where min_rows_cols =
        min(n_rows, n_cols), with the first entry containing the matched row indices, and the
        second the matched column indices.
    """
    import scipy

    cost_matrix_cpu = cost_matrix.detach().cpu()

    rows, cols = [], []
    for elem in cost_matrix_cpu:
        row_idxs, col_idxs = scipy.optimize.linear_sum_assignment(elem, maximize=maximize)
        rows.append(row_idxs)
        cols.append(col_idxs)

    row_idxs = torch.as_tensor(np.array(rows), dtype=torch.int64, device=cost_matrix.device)
    col_idxs = torch.as_tensor(np.array(cols), dtype=torch.int64, device=cost_matrix.device)

    return row_idxs, col_idxs


def _check_shape(x: torch.Tensor, expected_shape: Sequence[Optional[int]], name: str):
    """Verify shape of x is as expected.

    Adapted from SAVi Jax implementation:
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py
    """
    if not isinstance(expected_shape, (list, tuple)):
        raise ValueError(
            "`expected_shape` should be a list or tuple of ints but got " f"{expected_shape}."
        )

    # Scalars have shape () by definition.
    shape = getattr(x, "shape", ())

    if len(shape) != len(expected_shape) or any(
        j is not None and i != j for i, j in zip(shape, expected_shape)
    ):
        raise ValueError(f"Input {name} has shape {shape}, but expected {expected_shape}.")


# ---------------------------------------------------------------------------
# HOTA (Higher Order Tracking Accuracy) / AssA / DetA
# ---------------------------------------------------------------------------
# Reference:
#   Luiten et al., "HOTA: A Higher Order Metric for Evaluating Multi-Object
#   Tracking", IJCV 2021 (PMC7881978).
#
# This is a minimal, self-contained implementation that operates on
# per-video one-hot slot masks (predicted) and per-video one-hot GT masks
# and produces identity-aware tracking scores. It matches the structure of
# the official TrackEval reference implementation (averaging over
# localisation thresholds alpha in 0.05..0.95), but is written from scratch
# here because TrackEval is not available in the project environment.
# ---------------------------------------------------------------------------


class HigherOrderTrackingAccuracy(Metric):
    """Abstract HOTA / AssA / DetA metric.

    The metric is accumulated on a per-video basis. ``compute`` returns a
    dict with three scalars (``hota``, ``deta``, ``assa``) so the existing
    logging machinery (see :class:`JandFMetric`) emits
    ``val/<metric_name>/{hota,deta,assa}``.
    """

    higher_is_better = True
    full_state_update = False

    # Same alpha grid as the official TrackEval HOTA reference:
    # 0.05, 0.10, ..., 0.95 (19 thresholds).
    _DEFAULT_ALPHAS: Tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        alphas: Optional[Sequence[float]] = None,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(input_mapping={"pred_mask": pred_key, "true_mask": true_key})
        self.ignore_background = ignore_background
        self.ignore_overlaps = ignore_overlaps
        self.alphas = tuple(alphas) if alphas is not None else self._DEFAULT_ALPHAS
        n = len(self.alphas)
        # State: one running scalar per alpha for HOTA, DetA, AssA, plus a
        # count of videos contributing. We reduce by summation over DDP then
        # normalise in ``compute``.
        self.add_state(
            "hota_sum",
            default=torch.zeros(n, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "deta_sum",
            default=torch.zeros(n, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "assa_sum",
            default=torch.zeros(n, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    # ``_update`` is called with shape (batch, n_frames, n_true/pred_classes,
    # height, width) - see the per-video mixin below, which enforces this.
    def _update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        assert true_mask.ndim == 5 and pred_mask.ndim == 5
        if torch.any((true_mask != 0.0) & (true_mask != 1.0)):
            raise ValueError("`true_mask` is not binary")
        if torch.any((pred_mask != 0.0) & (pred_mask != 1.0)):
            raise ValueError("`pred_mask` is not binary")
        if torch.any(pred_mask.sum(dim=2) != 1.0):
            raise ValueError("`pred_mask` is not one-hot")

        n_true_classes_per_point = true_mask.sum(dim=2)
        if not self.ignore_overlaps and torch.any(n_true_classes_per_point > 1.0):
            raise ValueError("There are overlaps in `true_mask`.")
        if self.ignore_overlaps:
            overlaps = n_true_classes_per_point > 1.0
            true_mask = true_mask.clone()
            m = einops.repeat(overlaps, "b t h w -> b t c h w", c=true_mask.shape[2])
            true_mask[m] = 0.0
            pred_mask = pred_mask.clone()
            m = einops.repeat(overlaps, "b t h w -> b t k h w", k=pred_mask.shape[2])
            pred_mask[m] = 0.0

        if self.ignore_background:
            true_mask = true_mask[:, :, 1:]

        batch_size = true_mask.shape[0]
        for b in range(batch_size):
            # Skip videos without any ground truth (matches behaviour of
            # ARI/IoU metrics above).
            if true_mask[b].sum() == 0:
                continue
            scores = _video_hota_scores(
                true_mask[b].to(torch.bool),
                pred_mask[b].to(torch.bool),
                self.alphas,
            )
            if scores is None:
                continue
            hota_a, deta_a, assa_a = scores
            self.hota_sum += hota_a
            self.deta_sum += deta_a
            self.assa_sum += assa_a
            self.total += 1

    def compute(self):
        total = self.total.clamp(min=1)
        hota_per_alpha = self.hota_sum / total
        deta_per_alpha = self.deta_sum / total
        assa_per_alpha = self.assa_sum / total
        return {
            "hota": hota_per_alpha.mean(),
            "deta": deta_per_alpha.mean(),
            "assa": assa_per_alpha.mean(),
        }


class VideoHOTA(HigherOrderTrackingAccuracy):
    """HOTA metric for videos.

    Inputs to metric:
        true_mask: Binary true masks of shape (batch, n_frames, n_true_classes, height, width).
        pred_mask: One-hot predicted masks of shape (batch, n_frames, n_pred_classes, height,
            width).

    Args:
        ignore_background: If true, assume first channel of true masks is background to ignore.
        ignore_overlaps: If true, ignore pixels from overlapping ground-truth instances.
        alphas: Sequence of IoU thresholds to integrate over. Defaults to
            the TrackEval reference grid (0.05, 0.10, ..., 0.95).
    """

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        alphas: Optional[Sequence[float]] = None,
        pred_key: Optional[str] = None,
        true_key: Optional[str] = None,
    ):
        super().__init__(
            ignore_background=ignore_background,
            ignore_overlaps=ignore_overlaps,
            alphas=alphas,
            pred_key=pred_key,
            true_key=true_key,
        )


class VideoAssA(HigherOrderTrackingAccuracy):
    """AssA (association accuracy) sub-component of HOTA for videos.

    Returns only the association accuracy as a scalar so it can be logged
    as a single number. ``VideoHOTA`` already exposes AssA as a sub-key,
    so this class is primarily a convenience for configs that want AssA
    alone.
    """

    def compute(self):
        total = self.total.clamp(min=1)
        return (self.assa_sum / total).mean()


def _video_hota_scores(
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    alphas: Sequence[float],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compute HOTA / DetA / AssA for a single video across alphas.

    Args:
        true_mask: bool tensor of shape (n_frames, n_true_classes, h, w).
            Non-present GT tracks are simply all-zero along their channel.
        pred_mask: bool tensor of shape (n_frames, n_pred_classes, h, w),
            one-hot per pixel.
        alphas: sequence of IoU thresholds.

    Returns:
        Tuple of (hota, deta, assa), each of shape (len(alphas),), or None
        if the video is empty.
    """
    import scipy.optimize

    device = true_mask.device
    n_frames, n_true, h, w = true_mask.shape
    _, n_pred, _, _ = pred_mask.shape

    # Identify which GT / pred tracks are actually active anywhere in this
    # video. A track that never appears should not contribute to the
    # detection / association counts.
    true_flat = true_mask.reshape(n_frames, n_true, -1)  # (T, C, HW)
    pred_flat = pred_mask.reshape(n_frames, n_pred, -1)  # (T, K, HW)

    true_area_per_frame = true_flat.sum(-1).to(torch.float64)  # (T, C)
    pred_area_per_frame = pred_flat.sum(-1).to(torch.float64)  # (T, K)

    gt_active = true_area_per_frame.sum(0) > 0  # (C,)
    pred_active = pred_area_per_frame.sum(0) > 0  # (K,)
    if not torch.any(gt_active):
        return None
    n_true_active = int(gt_active.sum().item())
    n_pred_active = int(pred_active.sum().item())
    if n_pred_active == 0:
        # All GT tracks are FNs; every alpha scores 0.
        zeros = torch.zeros(len(alphas), dtype=torch.float64, device=device)
        return zeros, zeros, zeros

    true_idx_map = torch.nonzero(gt_active, as_tuple=False).flatten()
    pred_idx_map = torch.nonzero(pred_active, as_tuple=False).flatten()

    true_flat = true_flat[:, true_idx_map]  # (T, C', HW)
    pred_flat = pred_flat[:, pred_idx_map]  # (T, K', HW)
    true_area_per_frame = true_area_per_frame[:, true_idx_map]
    pred_area_per_frame = pred_area_per_frame[:, pred_idx_map]

    # Per-frame pairwise IoU. Computed once and reused for all alphas.
    # intersect[t, c, k] = |GT_c ∩ PR_k| in frame t.
    intersect = torch.einsum(
        "tcp,tkp->tck", true_flat.to(torch.float64), pred_flat.to(torch.float64)
    )  # (T, C', K')
    union = (
        true_area_per_frame.unsqueeze(-1)
        + pred_area_per_frame.unsqueeze(-2)
        - intersect
    )
    per_frame_iou = torch.where(union > 0, intersect / union, torch.zeros_like(union))

    # "Potential matches" across the whole video, used as a tie-break /
    # weighting term inside the Hungarian cost (matches TrackEval).
    # potential[c,k] = sum_t intersect[t,c,k] / max(gt_area_total[c], pr_area_total[k])
    gt_total = true_area_per_frame.sum(0)  # (C',)
    pr_total = pred_area_per_frame.sum(0)  # (K',)
    intersect_total = intersect.sum(0)  # (C', K')
    denom_total = torch.maximum(gt_total.unsqueeze(-1), pr_total.unsqueeze(-2))
    potential = torch.where(
        denom_total > 0,
        intersect_total / denom_total,
        torch.zeros_like(denom_total),
    )
    potential_np = potential.detach().cpu().numpy()

    per_frame_iou_np = per_frame_iou.detach().cpu().numpy()
    gt_presence = (true_area_per_frame > 0).detach().cpu().numpy()
    pr_presence = (pred_area_per_frame > 0).detach().cpu().numpy()

    hota_out = torch.zeros(len(alphas), dtype=torch.float64, device=device)
    deta_out = torch.zeros(len(alphas), dtype=torch.float64, device=device)
    assa_out = torch.zeros(len(alphas), dtype=torch.float64, device=device)

    # GT and pred frame counts (for FNA / FPA computation).
    gt_presence_count = gt_presence.sum(axis=0).astype(np.float64)  # (C',)
    pr_presence_count = pr_presence.sum(axis=0).astype(np.float64)  # (K',)

    for a_idx, alpha in enumerate(alphas):
        # Per-frame bipartite matching at the given alpha. Cost matrix is
        # -(1000 * IoU_valid + potential): IoU dominates the assignment,
        # potential breaks ties across frames so that a pred track tends
        # to stay bound to the same GT track when multiple are equally
        # matchable. Pairs below the alpha threshold are also retained in
        # the cost matrix (so Hungarian still runs) but filtered out of
        # the matches set afterwards.
        tp = 0
        tpa = np.zeros((n_true_active, n_pred_active), dtype=np.float64)
        match_count_gt = np.zeros(n_true_active, dtype=np.float64)
        match_count_pr = np.zeros(n_pred_active, dtype=np.float64)
        for t in range(n_frames):
            gt_here = np.where(gt_presence[t])[0]
            pr_here = np.where(pr_presence[t])[0]
            if gt_here.size == 0 or pr_here.size == 0:
                continue
            iou_t = per_frame_iou_np[t][np.ix_(gt_here, pr_here)]  # (g, p)
            score = 1000.0 * (iou_t >= alpha).astype(np.float64) * iou_t
            score = score + potential_np[np.ix_(gt_here, pr_here)]
            # Maximise score  -> minimise negative score.
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(-score)
            for r, c in zip(row_ind, col_ind):
                if iou_t[r, c] < alpha:
                    continue
                g = int(gt_here[r])
                p = int(pr_here[c])
                tp += 1
                tpa[g, p] += 1.0
                match_count_gt[g] += 1.0
                match_count_pr[p] += 1.0

        fn = int(gt_presence_count.sum() - tp)
        fp = int(pr_presence_count.sum() - tp)

        # Detection accuracy at alpha.
        det_denom = tp + fn + fp
        deta = tp / det_denom if det_denom > 0 else 0.0

        # Association accuracy at alpha: average over TPs of
        # TPA / (TPA + FNA + FPA), where FNA/FPA are derived from global
        # match counts per GT and per pred track.
        if tp == 0:
            assa = 0.0
        else:
            # Vectorised accumulation: for each (g, p) with tpa > 0,
            # FNA[g,p] = match_count_gt[g] - tpa[g,p]
            # FPA[g,p] = match_count_pr[p] - tpa[g,p]
            # contribution = tpa[g,p] * tpa[g,p] / (tpa[g,p] + FNA + FPA)
            #              = tpa[g,p]^2 / (match_count_gt[g] + match_count_pr[p] - tpa[g,p])
            mask_gp = tpa > 0
            if not mask_gp.any():
                assa = 0.0
            else:
                denom_gp = (
                    match_count_gt[:, None]
                    + match_count_pr[None, :]
                    - tpa
                )
                contribution = np.where(
                    (mask_gp) & (denom_gp > 0),
                    (tpa ** 2) / np.maximum(denom_gp, 1e-12),
                    0.0,
                )
                assa = contribution.sum() / tp

        hota = math.sqrt(max(deta, 0.0) * max(assa, 0.0))
        hota_out[a_idx] = hota
        deta_out[a_idx] = deta
        assa_out[a_idx] = assa

    return hota_out, deta_out, assa_out
