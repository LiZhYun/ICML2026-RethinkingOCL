import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tvt
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from slotcontrast import configuration, models
from slotcontrast.data.transforms import build_inference_transform
from slotcontrast.visualizations import color_map, draw_segmentation_masks_on_image
from data.ytvis import YTVOS


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, device: str = 'cuda'):
    """Load model from checkpoint with config."""
    config = configuration.load_config(config_path)
    model = models.build(config.model, config.optimizer)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    model.to(device)
    return model, config


def load_video_with_gt_masks(
    video_dir: Path, 
    video_id: str,
    ytvis: Optional[YTVOS],
    transform_config,
    device: str = 'cuda'
) -> Optional[Dict]:
    """Load video frames and optionally ground truth masks.
    
    Args:
        video_dir: Directory containing {frame_num:05d}.jpg files
        video_id: Video ID
        ytvis: YTVOS dataset instance (optional, for GT masks)
        transform_config: Transform configuration
        device: Device to use
        
    Returns:
        Dict with 'video', 'video_visualization', and optionally 'segmentations' keys
    """
    # Load frames
    frame_files = sorted([f for f in video_dir.glob("*.jpg")])
    if len(frame_files) == 0:
        return None
    
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert('RGB')
        img_array = np.array(img)
        frames.append(img_array)
    
    # Stack to video: [T, H, W, 3]
    video = np.stack(frames, axis=0)
    T, H_orig, W_orig, _ = video.shape
    video = torch.from_numpy(video).float() / 255.0
    
    # Prepare visualization: resize to input_size
    video_vis = video.permute(0, 3, 1, 2)  # [T, 3, H, W]
    video_vis = tvt.Resize((transform_config.input_size, transform_config.input_size))(video_vis)
    video_vis = video_vis.permute(1, 0, 2, 3)  # [3, T, H, W]
    
    # Apply transforms for model input
    if transform_config:
        tfs = build_inference_transform(transform_config)
        video = video.permute(3, 0, 1, 2)  # [3, T, H, W]
        video = tfs(video).permute(1, 0, 2, 3)  # [T, 3, H, W]
    else:
        video = video.permute(0, 3, 1, 2)  # [T, 3, H, W]
    
    # Load ground truth masks if YTVIS annotations available
    gt_masks = None
    if ytvis is not None:
        vid_info = None
        for vid in ytvis.vids.values():
            if vid['file_names'][0].split('/')[0] == video_id:
                vid_info = vid
                break
        
        if vid_info is not None:
            vid_id = vid_info['id']
            ann_ids = ytvis.getAnnIds(vidIds=[vid_id])
            anns = ytvis.loadAnns(ann_ids)
            
            # Decode masks for all frames and objects
            # GT format: [T, K, H, W] where K is number of objects
            gt_masks_list = []
            for frame_idx in range(T):
                frame_masks = []
                for ann in anns:
                    if ann['segmentations'][frame_idx] is not None:
                        mask = ytvis.annToMask(ann, frame_idx)
                        frame_masks.append(mask)
                
                if len(frame_masks) > 0:
                    # Stack masks for this frame: [K, H, W]
                    frame_masks = np.stack(frame_masks, axis=0)
                else:
                    # No objects in this frame, create empty mask
                    frame_masks = np.zeros((1, H_orig, W_orig), dtype=np.uint8)
                
                gt_masks_list.append(frame_masks)
            
            # Stack to [T, K, H, W] - note K may vary per frame
            # We need to pad to max K across all frames
            max_objects = max(len(fm) for fm in gt_masks_list)
            padded_masks = []
            for frame_masks in gt_masks_list:
                K_frame = frame_masks.shape[0]
                if K_frame < max_objects:
                    padding = np.zeros((max_objects - K_frame, H_orig, W_orig), dtype=np.uint8)
                    frame_masks = np.concatenate([frame_masks, padding], axis=0)
                padded_masks.append(frame_masks)
            
            gt_masks = np.stack(padded_masks, axis=0)  # [T, K, H, W]
            gt_masks = torch.from_numpy(gt_masks).float()
            
            # Resize GT masks to match model input size
            # Reshape to [T*K, 1, H, W] for interpolation
            T_m, K, H, W = gt_masks.shape
            gt_masks = gt_masks.reshape(T_m * K, 1, H, W)
            gt_masks = F.interpolate(
                gt_masks, 
                size=(transform_config.input_size, transform_config.input_size),
                mode='nearest'
            )
            gt_masks = gt_masks.reshape(T_m, K, transform_config.input_size, transform_config.input_size)
            
            # Permute to [K, T, H, W] for metrics (standard format)
            gt_masks = gt_masks.permute(1, 0, 2, 3)
    
    # Add batch dimension
    inputs = {
        "video": video.unsqueeze(0).to(device),  # [1, T, 3, H, W]
        "video_visualization": video_vis.unsqueeze(0).to(device),  # [1, 3, T, H, W]
    }
    
    if gt_masks is not None:
        inputs["segmentations"] = gt_masks.unsqueeze(0).to(device)  # [1, K, T, H, W]
    
    return inputs


def visualize_and_save_predictions(
    inputs: Dict,
    pred_masks: torch.Tensor,
    output_dir: Path,
    video_id: str,
    n_slots: int,
    alpha: float = 0.5,
):
    """Save predicted mask visualizations for a video.
    
    Args:
        inputs: Input dict with 'video_visualization'
        pred_masks: [1, K, T, H, W] predicted masks
        output_dir: Output directory for this video
        video_id: Video ID
        n_slots: Number of slots
        alpha: Transparency for mask overlay
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get visualization frames: [1, 3, T, H, W] -> [T, 3, H, W]
    video_vis = inputs["video_visualization"][0].permute(1, 0, 2, 3).cpu()
    
    # Get masks: [1, K, T, H, W] -> [T, K, H, W]
    masks = pred_masks[0].permute(1, 0, 2, 3).cpu()
    
    T = video_vis.shape[0]
    
    # Get color map for all slots
    colors = color_map(n_slots)
    
    for t in range(T):
        frame = video_vis[t]  # [3, H, W]
        frame_masks = masks[t]  # [K, H, W]
        
        # Convert to uint8
        frame_uint8 = (frame * 255).byte()
        
        # Draw masks
        vis_frame = draw_segmentation_masks_on_image(
            frame_uint8,
            frame_masks.bool(),
            colors=colors,
            alpha=alpha
        )
        
        # Convert to numpy and save
        vis_frame_np = vis_frame.permute(1, 2, 0).numpy()
        imageio.imwrite(output_dir / f"frame_{t:05d}.png", vis_frame_np)


def process_video(
    video_dir: Path,
    video_id: str,
    model,
    ytvis: Optional[YTVOS],
    transform_config,
    output_dir: Path,
    n_slots: int,
    device: str = 'cuda',
) -> Optional[Dict]:
    """Process single video: load data, run inference, save visualizations.
    
    Returns:
        Dict with video_id and info, or None if failed
    """
    # Load video with GT masks
    inputs = load_video_with_gt_masks(
        video_dir, video_id, ytvis, transform_config, device
    )
    
    if inputs is None:
        return None
    
    T = inputs["video"].shape[1]
    print(f"Processing {video_id}: {T} frames")
    
    # Run inference
    with torch.no_grad():
        outputs = model(inputs)
        aux_outputs = model.aux_forward(inputs, outputs)
    
    # Get hard masks: [1, T, K, H, W] -> [1, K, T, H, W]
    pred_masks = aux_outputs["decoder_masks_hard"].permute(0, 2, 1, 3, 4)
    
    # Save visualizations
    video_output_dir = output_dir / video_id
    visualize_and_save_predictions(
        inputs, pred_masks, video_output_dir, video_id, n_slots
    )
    
    # Save per-video info
    results = {
        "video_id": video_id,
        "n_frames": T,
        "n_pred_slots": n_slots,
    }
    
    # Add GT object count if available
    if "segmentations" in inputs:
        results["n_true_objects"] = inputs["segmentations"].shape[1]
    
    with open(video_output_dir / "info.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


def save_aggregate_results(
    all_results: List[Dict],
    output_dir: Path,
    checkpoint_name: str,
):
    """Save aggregate results as CSV and summary statistics as JSON."""
    # Sort by video_id
    all_results_sorted = sorted(all_results, key=lambda x: x["video_id"])
    
    # Save as CSV
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["video_id", "n_frames", "n_true_objects", "n_pred_slots"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results_sorted)
    
    # Compute summary statistics
    n_frames_list = [r["n_frames"] for r in all_results]
    
    summary = {
        "checkpoint": checkpoint_name,
        "n_videos": len(all_results),
        "total_frames": sum(n_frames_list),
        "avg_frames_per_video": float(np.mean(n_frames_list)),
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results Summary for {checkpoint_name}")
    print(f"{'='*60}")
    print(f"Videos processed: {summary['n_videos']}")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Avg frames per video: {summary['avg_frames_per_video']:.1f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - {csv_path.name}")
    print(f"  - summary.json")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch inference with metrics for YTVIS2021 validation")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--data-dir",
        default="/data/ytvis2021_raw/valid",
        help="Path to validation directory (e.g., ytvis2021_raw/valid, movi_d_raw/valid, movi_e_raw/valid)"
    )
    parser.add_argument(
        "--output-dir",
        default="/data/ytvis2021_inference",
        help="Output directory for inference results"
    )
    parser.add_argument(
        "--n-slots",
        type=int,
        default=7,
        help="Number of slots for inference"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (for testing)"
    )
    parser.add_argument(
        "--video-ids",
        nargs="+",
        help="Specific video IDs to process (e.g., 00f88c4f0a 01c88b5b60)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    # Extract checkpoint name for output organization
    checkpoint_path = Path(args.checkpoint)
    checkpoint_name = checkpoint_path.parent.parent.name  # e.g., greedy_per_no_predictor_ytvis_3iter
    
    # Setup paths
    data_dir = Path(args.data_dir)
    jpeg_dir = data_dir / "JPEGImages"
    anno_file = data_dir / "instances.json"
    
    output_dir = Path(args.output_dir) / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations if available (YTVIS), otherwise skip (MOVI)
    ytvis = None
    if anno_file.exists():
        print(f"Loading annotations from {anno_file}...")
        ytvis = YTVOS(str(anno_file))
    else:
        print(f"No annotations file found at {anno_file}")
        print(f"Running inference without GT masks (MOVI mode)...")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model_from_checkpoint(args.checkpoint, args.config, args.device)
    model.initializer.n_slots = args.n_slots
    
    # Create transform config
    transform_config = OmegaConf.create({
        'dataset_type': 'video',
        'input_size': config.dataset.get('input_size', 224)
    })
    
    # Get video directories
    if args.video_ids:
        video_dirs = [jpeg_dir / vid for vid in args.video_ids if (jpeg_dir / vid).exists()]
    else:
        video_dirs = sorted([d for d in jpeg_dir.iterdir() if d.is_dir()])
    
    if args.max_videos:
        video_dirs = video_dirs[:args.max_videos]
    
    print(f"Processing {len(video_dirs)} videos...")
    print(f"Output directory: {output_dir}")
    
    # Process videos
    all_results = []
    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        video_id = video_dir.name
        try:
            results = process_video(
                video_dir,
                video_id,
                model,
                ytvis,
                transform_config,
                output_dir,
                args.n_slots,
                args.device,
            )
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save aggregate results
    if all_results:
        save_aggregate_results(all_results, output_dir, checkpoint_name)
    else:
        print("No results to save!")
    
    print(f"\nDone! Processed {len(all_results)} videos successfully.")


if __name__ == "__main__":
    main()
