"""Script to download and shard YouTube VIS dataset."""
import argparse
import os
import sys

import einops
import numpy as np
import torch
import tqdm
import webdataset as wds
from einops import rearrange
from PIL import Image
from pathlib import Path
from data.utils import get_default_dir
from data.ytvis import YTVOS as YTVIS

from slotcontrast.data.transforms import Resize
from slotcontrast.data.transforms_video import FromTensorVideo, ToTensorVideo

# Add TAPIP3D to path for depth/pose estimation
TAPIP3D_PATH = Path(__file__).parent.parent.parent / "TAPIP3D"
sys.path.insert(0, str(TAPIP3D_PATH))
from annotation.megasam import MegaSAMAnnotator
import cv2

# TODO: add download from google drive
# for now use gdrive_download script with IDs above.
# TRAIN_GDRIVE_ID = "1fLO_d8ys_03CPPpCm-lL5SgCJsbNN5L9"
# VAL_GDRIVE_ID = "1Z0VEOQp4d0-aiZNDt8Hd5KdcCU4s8JaY"

SPLITS_TO_SUFFIX = {
    "train": "train",
    "validation": "valid",
}
SPLITS_TO_INSTANCES_FILENAME = {
    "train": "instances.json",
    "validation": "instances.json",
}


parser = argparse.ArgumentParser("Generate sharded dataset from original YouTube-VIS data.")
parser.add_argument(
    "--split", default="train", choices=list(SPLITS_TO_SUFFIX), help="Which splits to write"
)
parser.add_argument(
    "--download-dir",
    default=get_default_dir("ytvis2021_raw"),
    help="Directory where YTVIS 2021 videos and annotations are downloaded to",
)
parser.add_argument("--maxcount", type=int, default=32, help="Max number of samples per shard")
parser.add_argument("--only-videos", action="store_true", help="Whether to store only videos")
parser.add_argument("--resize", action="store_true", help="Whether to resize videos")
parser.add_argument(
    "--resize_size", type=int, default=320, help="The size of min(H,W) after resizing"
)
parser.add_argument("--resize_mode", type=str, default="nearest-exact", help="Resize mode")
parser.add_argument(
    "--out-path",
    default=get_default_dir("ytvis2021_resized2"),
    help="Directory where shards are written",
)
parser.add_argument(
    "--add-depth", action="store_true", help="Whether to compute and add depth maps and camera poses"
)
parser.add_argument(
    "--depth-model", type=str, default="moge", choices=["dav1", "dav2", "videoda", "moge"],
    help="Depth model to use for MegaSAM"
)
parser.add_argument(
    "--device", type=str, default="cuda", help="Device for depth estimation"
)
parser.add_argument(
    "--save-depth-vis", type=int, default=0, 
    help="Number of depth visualization videos to save (0 to disable)"
)


import matplotlib.pyplot as plt


def save_depth_video(video, depths, out_path, vid_id, fps=10):
    """Save side-by-side RGB and depth visualization video."""
    import matplotlib.cm as cm
    h, w = video.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_file = os.path.join(out_path, f"depth_vis_{vid_id}.mp4")
    writer = cv2.VideoWriter(out_file, fourcc, fps, (w * 2, h))
    
    # Normalize depths for visualization
    d_min, d_max = depths.min(), depths.max()
    depths_norm = (depths - d_min) / (d_max - d_min + 1e-8)
    
    for frame, depth in zip(video, depths_norm):
        # Apply colormap to depth
        depth_colored = (cm.plasma(depth)[:, :, :3] * 255).astype(np.uint8)
        # Concatenate RGB and depth side by side (convert RGB to BGR for cv2)
        combined = np.concatenate([frame[:, :, ::-1], depth_colored[:, :, ::-1]], axis=1)
        writer.write(combined)
    writer.release()
    print(f"Saved depth visualization: {out_file}")


def write_dataset(video_dir, out_dir, split, add_annotations=True, annotations_file=None, 
                  megasam_annotator=None, save_depth_vis=0):
    if add_annotations and annotations_file is None:
        raise ValueError("Requested to add annotations but annotations file not given")

    if not os.path.isdir(os.path.join(out_dir, ".")):
        os.makedirs(os.path.join(out_dir, "."), exist_ok=True)

    # This is the output pattern under which we write shards.
    pattern = os.path.join(out_dir, f"ytvis-{split}-%06d.tar")
    if args.resize:
        clamp_zero_one = args.resize_mode == "bicubic"
        resize = Resize(
            args.resize_size, args.resize_mode, clamp_zero_one=clamp_zero_one, short_side_scale=True
        )
        segmentation_resize = Resize(
            args.resize_size, "nearest-exact", clamp_zero_one=False, short_side_scale=True
        )
        to_video = ToTensorVideo()
        from_video = FromTensorVideo()
    ytvis = YTVIS(annotations_file)
    all_ids = ytvis.getVidIds()
    val_ids = all_ids[-300:]
    train_ids = all_ids[:-300]
    ids = train_ids if split == "train" else val_ids
    print(f"Saving {len(ids)} from {len(all_ids)} as a {split} part of the datasets.")
    depth_vis_count = 0
    with wds.ShardWriter(pattern, maxcount=args.maxcount) as sink:
        max_num_instances = 0
        for vid_id in tqdm.tqdm(ids):
            vid = ytvis.vids[vid_id]
            length = int(vid["length"])
            height = int(vid["height"])
            width = int(vid["width"])
            video = []

            for _, file_name in enumerate(vid["file_names"]):
                frame_path = os.path.join(video_dir, file_name)
                with Image.open(frame_path) as img:
                    img = np.asarray(img, dtype="uint8")
                video.append(img[None])
            video = np.concatenate(video)
            # video shape should be [B, F, H, W, 3]
            assert video.shape == (length, height, width, 3)
            if args.resize:
                video = to_video(torch.from_numpy(video))
                video = from_video(resize(video)).cpu().numpy()

            # Compute depth maps and camera poses if requested
            depths, intrinsics, extrinsics = None, None, None
            if megasam_annotator is not None:
                # Get video resolution for depth processing
                target_h, target_w = video.shape[1:3]
                
                # Process video to get depth, intrinsics, extrinsics
                # return_raw_depths=True returns depths at MegaSAM's internal resolution
                raw_depths, intrinsics, extrinsics = megasam_annotator.process_video(
                    video, gt_intrinsics=None, return_raw_depths=True
                )
                
                # Resize raw depths to match video resolution (following TAPIP3D inference.py)
                depths = np.stack([
                    cv2.resize(d.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    for d in raw_depths
                ])
                
                # Save depth visualization for first N videos
                if depth_vis_count < save_depth_vis:
                    save_depth_video(video, depths, out_dir, vid_id)
                    depth_vis_count += 1

            if not add_annotations:
                sample = {
                    "__key__": str(vid_id),
                    "video.npy": video,
                }
                if depths is not None:
                    sample["depths.npy"] = depths
                    sample["intrinsics.npy"] = intrinsics
                    sample["extrinsics.npy"] = extrinsics
            else:

                segmentations = []
                ann_ids = ytvis.getAnnIds(vidIds=[vid_id], iscrowd=False)
                anns = ytvis.loadAnns(ann_ids)
                if anns:
                    for f_id, _ in enumerate(vid["file_names"]):
                        seg = []
                        num_obj = len(anns)
                        max_num_instances = max(max_num_instances, num_obj)
                        for ann in anns:
                            try:
                                mask = ytvis.annToMask(ann, f_id) * ann["category_id"]
                            except TypeError:
                                # TODO: check if this is the right way to do this.
                                # e.g. that error actually means that instance
                                # is not visible in the image
                                mask = np.zeros(
                                    (int(vid["height"]), int(vid["width"])), dtype=np.uint8
                                )
                            seg.append(mask)
                        segmentations.append(seg)
                    segmentations = np.asarray(segmentations, dtype=np.uint8)
                    segmentations = rearrange(segmentations, "f i h w -> f h w i")

                    # segmentation shape should be [B, F, H, W, 1]
                    assert segmentations.shape == (
                        length,
                        height,
                        width,
                        num_obj,
                    )
                else:
                    num_obj = 0
                    segmentations = np.zeros(
                        (length, int(vid["height"]), int(vid["width"]), 1), dtype=np.uint8
                    )
                if args.resize:
                    segmentations = torch.from_numpy(segmentations)
                    segmentations = einops.rearrange(segmentations, "f h w c -> f c h w")
                    segmentations = segmentation_resize(segmentations).cpu().numpy()
                    segmentations = einops.rearrange(segmentations, "f c h w -> f h w c")

                # Pad instance channel to fixed cap of 32 so all shards have a
                # consistent shape for batched collation. Per-annotation binary
                # masks are preserved as channels 0..num_obj-1; remaining channels
                # are zero-filled. Compact local track IDs (1..num_obj) live in
                # `track_ids.npy`, padded with 0 for the absent slots.
                CAP = 32
                cur_f, cur_h, cur_w, cur_n = segmentations.shape
                if cur_n < CAP:
                    pad = np.zeros((cur_f, cur_h, cur_w, CAP - cur_n), dtype=segmentations.dtype)
                    segmentations = np.concatenate([segmentations, pad], axis=-1)
                elif cur_n > CAP:
                    # Truncate; large-K_video videos lose their tail tracks.
                    # Logged via max_num_instances above; fixed cap is required for batched collate.
                    segmentations = segmentations[..., :CAP]
                track_ids = np.zeros((CAP,), dtype=np.int32)
                kept = min(num_obj, CAP)
                track_ids[:kept] = np.arange(1, kept + 1, dtype=np.int32)
                sample = {
                    "__key__": str(vid_id),
                    "video.npy": video,
                    "segmentations.npy": segmentations,
                    "track_ids.npy": track_ids,
                }
                if depths is not None:
                    sample["depths.npy"] = depths
                    sample["intrinsics.npy"] = intrinsics
                    sample["extrinsics.npy"] = extrinsics
            sink.write(sample)

    if add_annotations:
        print(f"Maximal number of instances per video in {split} split is: {max_num_instances}.")


if __name__ == "__main__":
    args = parser.parse_args()
    root = os.path.join(args.download_dir, "train")
    annotations_file = os.path.join(root, "instances.json")
    video_dir = os.path.join(root, "JPEGImages")
    
    # Initialize MegaSAM annotator if depth estimation is requested
    megasam_annotator = None
    if args.add_depth:
        # Resolution should be height * width (not squared resize_size)
        # Use a reasonable default resolution for MegaSAM internal processing
        resolution = args.resize_size * args.resize_size if args.resize else 320 * 480
        megasam_annotator = MegaSAMAnnotator(
            script_path=TAPIP3D_PATH / "third_party" / "megasam" / "inference.py",
            depth_model=args.depth_model,
            resolution=resolution,
        )
        print(f"Initialized MegaSAMAnnotator with depth_model={args.depth_model}, resolution={resolution}")
    
    write_dataset(
        video_dir=video_dir,
        out_dir=args.out_path,
        split=args.split,
        add_annotations=annotations_file is not None and not args.only_videos,
        annotations_file=annotations_file,
        megasam_annotator=megasam_annotator,
        save_depth_vis=args.save_depth_vis,
    )
