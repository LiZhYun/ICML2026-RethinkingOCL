import abc
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import webdataset as wds

from slotcontrast.data import transforms
from slotcontrast.data.utils import remap_dict
from slotcontrast.utils import config_as_kwargs


def pack_camera_data(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Pack depth, intrinsics, extrinsics into camera_data dict if present."""
    if "depths" in sample and "intrinsics" in sample and "extrinsics" in sample:
        sample["camera_data"] = {
            "depth": sample.pop("depths"),
            "intrinsics": sample.pop("intrinsics"),
            "extrinsics": sample.pop("extrinsics"),
        }
    return sample


def build(config, name: Optional[str] = "VideoPipeline", **kwargs):
    name = config.get("name") or name
    if name in ("video", "VideoPipeline"):
        tfs = transforms.build(config.transforms) if config.transforms else None
        pipeline = VideoPipeline(
            transforms=tfs,
            **config_as_kwargs(config, to_filter=("transforms",), defaults=kwargs),
        )
    elif name in ("image", "ImagePipeline"):
        tfs = transforms.build(config.transforms) if config.transforms else None
        pipeline = ImagePipeline(
            transforms=tfs,
            **config_as_kwargs(config, to_filter=("transforms",), defaults=kwargs),
        )
    else:
        raise ValueError(f"Unknown pipeline `{name}`")

    return pipeline


class DataPipeline(abc.ABC):
    """Abstract base class for data pipelines.

    A data pipeline defines how a stream of raw data samples is transformed to the stream of
    output samples.
    """

    def __init__(self, keys: Optional[Tuple[str]]):
        self._keys = keys

    @property
    def keys(self) -> Optional[Tuple[str]]:
        """Keys of properties to keep in dataset after filtering."""
        return self._keys

    @abc.abstractmethod
    def get_num_samples(self, num_orig_samples: int) -> Optional[int]:
        """Number of samples after pipeline is applied, given original number of samples in dataset.

        If number can not be computed, may return `None`.
        """
        ...

    @abc.abstractmethod
    def apply(self, dataset: wds.WebDataset) -> wds.WebDataset:
        """Apply pipeline to dataset.

        Input dataset contains dicts of samples after decoding.
        """
        ...


class ImagePipeline(DataPipeline):
    def __init__(
        self,
        transforms: Dict[str, Optional[Callable]] = None,
        keys: Optional[Tuple[str]] = None,
        is_video_dataset: bool = True,
        video_size: Optional[int] = None,
        one_frame_per_video: bool = False,
        shuffle: bool = False,
        shuffle_size: int = 1000,
        duplicate: Optional[Dict[str, str]] = None,
    ):
        super().__init__(keys)
        self.transforms = transforms
        self.video_size = video_size
        self.shuffle = shuffle
        self.shuffle_size = shuffle_size
        self.is_video_dataset = is_video_dataset
        self.one_frame_per_video = one_frame_per_video
        self.duplicate = duplicate

    def get_num_samples(self, num_orig_samples: int) -> Optional[int]:
        if self.video_size is not None:
            return num_orig_samples * self.video_size
        else:
            return None  # Can not provide the number of samples

    def apply(self, dataset: wds.WebDataset) -> wds.WebDataset:
        if self.is_video_dataset:
            split_fn = partial(
                split_to_chunks,
                keys_to_split=self.keys,
                chunk_size=1,
                shuffle=self.shuffle,
                one_chunk_per_video=self.one_frame_per_video,
            )
            dataset = dataset.compose(split_fn)
            rename_video = partial(remap_dict, rename_dict={"video": "image"})
            dataset.map(rename_video)

        if self.shuffle:
            # First chunking, then shuffling
            dataset = dataset.shuffle(self.shuffle_size)

        if self.duplicate:
            dataset.map(partial(copy_dict_entries, copy_from_to=self.duplicate))

        if self.transforms is not None:
            dataset = dataset.map_dict(**self.transforms)

        return dataset


class VideoPipeline(DataPipeline):
    def __init__(
        self,
        transforms: Dict[str, Optional[Callable]] = None,
        keys: Optional[Tuple[str]] = None,
        video_size: Optional[int] = None,
        chunk_size: int = 6,
        use_chunks: bool = True,
        sample_one_chunk_per_video: bool = False,
        shuffle: bool = False,
        shuffle_size: int = 100,
        duplicate: Optional[Dict[str, str]] = None,
        frame_stride: int = 1,
        occl_curriculum: Optional[Dict] = None,
    ):
        super().__init__(keys)
        self.transforms = transforms
        self.video_size = video_size
        self.chunk_size = chunk_size
        self.use_chunks = use_chunks
        if sample_one_chunk_per_video and not use_chunks:
            raise ValueError("For sampling one chunk per video, use_chunks needs to be True")
        self.sample_one_chunk_per_video = sample_one_chunk_per_video
        self.shuffle = shuffle
        self.shuffle_size = shuffle_size
        self.duplicate = duplicate
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        self.frame_stride = frame_stride
        # Round-26 B2: config-gated OcclCurr hook. `occl_curriculum` is a dict
        # with keys {enabled, curriculum_steps, max_ell_target}; when enabled,
        # inject `OcclusionCurriculumSampler` after `split_to_chunks`.
        self.occl_curriculum = occl_curriculum or {}

    def get_num_samples(self, num_orig_samples: int) -> Optional[int]:
        if self.use_chunks:
            if self.sample_one_chunk_per_video:
                return num_orig_samples
            else:
                if self.video_size is not None:
                    # Each chunk spans `chunk_size * frame_stride` raw frames
                    span = self.chunk_size * self.frame_stride
                    return num_orig_samples * self.video_size // span
                else:
                    return None  # Can not provide the number of samples
        else:
            return num_orig_samples

    def apply(self, dataset: wds.WebDataset) -> wds.WebDataset:
        if self.use_chunks:
            # If sampling chunks, need to shuffle as well to pick a random chunk
            shuffle = self.shuffle or self.sample_one_chunk_per_video
            split_fn = partial(
                split_to_chunks,
                keys_to_split=self.keys,
                shuffle=shuffle,
                one_chunk_per_video=self.sample_one_chunk_per_video,
                chunk_size=self.chunk_size,
                frame_stride=self.frame_stride,
            )
            dataset = dataset.compose(split_fn)

        # Round-26 B2: OcclCurr hook applied after chunking, before shuffle.
        # No-op unless occl_curriculum.enabled=true in config.
        if self.use_chunks and self.occl_curriculum.get("enabled", False):
            from slotcontrast.data.transforms import OcclusionCurriculumSampler
            sampler = OcclusionCurriculumSampler(
                curriculum_steps=int(self.occl_curriculum.get("curriculum_steps", 25000)),
                max_ell_target=int(self.occl_curriculum.get("max_ell_target", 8)),
            )
            dataset = dataset.compose(sampler)

        if self.shuffle:
            # First chunking, then shuffling
            dataset = dataset.shuffle(self.shuffle_size)

        if self.duplicate:
            dataset.map(partial(copy_dict_entries, copy_from_to=self.duplicate))

        if self.transforms is not None:
            dataset = dataset.map_dict(**self.transforms)

        return dataset


def split_to_chunks(
    data,
    keys_to_split: Tuple[str],
    chunk_size: int,
    shuffle: bool,
    one_chunk_per_video: bool,
    axis: int = 0,
    frame_stride: int = 1,
):
    """Split video to chunks with chunk_size size.

    When `frame_stride > 1`, each chunk samples frames at positions
    `[t, t+stride, ..., t+(chunk_size-1)*stride]`. A chunk spans
    `chunk_size * frame_stride` raw frames; chunks are non-overlapping
    in the start-position grid (start positions: `0, span, 2*span, ...`).
    `frame_stride=1` reproduces the previous contiguous behavior.
    """
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    for sample in data:
        key = sample["__key__"]
        video_size = sample[keys_to_split[0]].shape[0]

        if frame_stride == 1:
            num_chunks = video_size // chunk_size
            data_chunks = [
                np.array_split(
                    sample[key],
                    range(chunk_size, video_size, chunk_size),
                    axis=axis,
                )[:num_chunks]
                for key in keys_to_split
            ]
        else:
            span = chunk_size * frame_stride
            num_chunks = video_size // span
            # For each chunk c, build index list [c*span, c*span + stride, ...,
            # c*span + (chunk_size-1)*stride] (length = chunk_size). Slicing on
            # `axis` with `np.take` keeps non-temporal dims intact.
            data_chunks = []
            for key in keys_to_split:
                arr = sample[key]
                per_key = []
                for c in range(num_chunks):
                    base = c * span
                    idx = np.arange(chunk_size, dtype=np.int64) * frame_stride + base
                    per_key.append(np.take(arr, idx, axis=axis))
                data_chunks.append(per_key)

        if shuffle:
            chunks_ids = np.random.permutation(range(num_chunks))
        else:
            chunks_ids = list(range(num_chunks))

        if one_chunk_per_video:
            chunks_ids = chunks_ids[:1]

        for chunk_id in chunks_ids:
            chunked_data = {
                key: data_key[chunk_id] for key, data_key in zip(keys_to_split, data_chunks)
            }
            for key, value in sample.items():
                if key in keys_to_split:
                    continue
                # Data that is not splitted is just repeated across all chunks
                chunked_data[key] = value
            chunked_data["__key__"] = f"{key}_{chunk_id}"
            yield chunked_data


def copy_dict_entries(dictionary: Dict[str, Any], copy_from_to: Dict[str, str]) -> Dict[str, Any]:
    for from_key, to_key in copy_from_to.items():
        dictionary[to_key] = dictionary[from_key]

    return dictionary
