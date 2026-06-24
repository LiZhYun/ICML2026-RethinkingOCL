#!/usr/bin/env python3
"""Extract MOVI validation data to YTVIS-like directory structure.

This script reads WebDataset tar files and extracts them to:
    movi_{d,e}_raw/valid/JPEGImages/{video_id}/{frame_id:05d}.jpg

Compatible with existing inference and visualization scripts.
"""

import argparse
import io
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import tarfile
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def extract_movi_validation(
    tar_files: list,
    output_dir: Path,
    max_videos: int = None,
):
    """Extract MOVI validation tar files to JPEG directory structure.
    
    Args:
        tar_files: List of paths to .tar files
        output_dir: Output directory (e.g., movi_d_raw/valid)
        max_videos: Maximum number of videos to extract
    """
    jpeg_dir = output_dir / "JPEGImages"
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create directory for segmentations if needed
    seg_dir = output_dir / "Segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)
    
    video_count = 0
    
    for tar_path in sorted(tar_files):
        print(f"\nProcessing {tar_path}...")
        
        with tarfile.open(tar_path, 'r') as tar:
            # Group files by sample key
            samples = {}
            for member in tar.getmembers():
                if member.isfile():
                    # Parse key from filename: {key}.{ext}
                    name = member.name
                    if '.' in name:
                        key, ext = name.rsplit('.', 1)
                        if key not in samples:
                            samples[key] = {}
                        samples[key][ext] = member
            
            # Process each sample
            for sample_key in tqdm(sorted(samples.keys()), desc=f"Extracting from {tar_path.name}"):
                if max_videos and video_count >= max_videos:
                    print(f"Reached max_videos limit ({max_videos})")
                    return
                
                sample = samples[sample_key]
                
                # Extract video frames
                if 'npy' in sample:
                    # Load the .npy file (could be video.npy or just {key}.npy)
                    member = sample['npy']
                    f = tar.extractfile(member)
                    data = np.load(io.BytesIO(f.read()))
                    
                    # Determine if this is video or segmentation
                    # video: [T, H, W, 3], uint8
                    # segmentations: [T, H, W], uint8 or [T, H, W, 1]
                    if data.ndim == 4 and data.shape[-1] == 3:
                        # This is video data
                        video_data = data
                        
                        # Create video directory
                        video_id = f"{video_count:04d}"
                        video_dir = jpeg_dir / video_id
                        video_dir.mkdir(exist_ok=True)
                        
                        # Save each frame as JPEG
                        T = video_data.shape[0]
                        for t in range(T):
                            frame = video_data[t]  # [H, W, 3]
                            frame_path = video_dir / f"{t:05d}.jpg"
                            
                            # Convert to PIL Image and save
                            img = Image.fromarray(frame)
                            img.save(frame_path, quality=95)
                        
                        video_count += 1
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Total videos extracted: {video_count}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract MOVI validation data to YTVIS-like structure")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["movi_c", "movi_d", "movi_e"],
        help="Which MOVI dataset to extract"
    )
    parser.add_argument(
        "--data-dir",
        default="/home/zhiyuan/Codes/exp_slotcontrast/data",
        help="Path to data directory containing tar files"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: {data-dir}/{dataset}_raw/valid)"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to extract (for testing)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    dataset_dir = data_dir / args.dataset
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / f"{args.dataset}_raw" / "valid"
    
    # Find all validation tar files
    tar_pattern = f"{args.dataset}-validation-*.tar"
    tar_files = sorted(dataset_dir.glob(tar_pattern))
    
    if not tar_files:
        print(f"Error: No tar files found matching {dataset_dir / tar_pattern}")
        sys.exit(1)
    
    print(f"Found {len(tar_files)} tar files:")
    for tar_file in tar_files:
        print(f"  - {tar_file.name}")
    
    # Extract
    extract_movi_validation(tar_files, output_dir, args.max_videos)


if __name__ == "__main__":
    main()
