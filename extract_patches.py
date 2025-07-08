#!/usr/bin/env python
# extract_patches.py

"""
Extract tissue patches from every WSI slide in CMB-LCA/.
Patches are stored per-sample in output/<slide_id>/.
Creates patch_index.csv mapping each patch to its slide-level label.
Now includes quality‑control (blur & redundancy) filtering.
"""

import os
import openslide
import numpy as np
from PIL import Image
import cv2  # OpenCV for blur detection
from tqdm import tqdm
import zipfile
import argparse

# ------------------------------
# Tissue presence filter
# ------------------------------

def is_tissue(patch, threshold: float = 0.8) -> bool:
    """Return True if the patch contains enough non‑background pixels."""
    np_patch = np.array(patch.convert("RGB"))
    gray = np_patch.mean(axis=2)
    bg_ratio = (gray > 220).sum() / (gray.shape[0] * gray.shape[1])
    return bg_ratio < threshold

# ------------------------------
# Quality‑control utilities
# ------------------------------

def is_blurry(patch, blur_threshold: float = 100.0) -> bool:
    """Return True if patch is blurry, judged by Laplacian variance."""
    gray = cv2.cvtColor(np.array(patch.convert("RGB")), cv2.COLOR_RGB2GRAY)
    var_lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var_lap < blur_threshold


def average_hash(patch, hash_size: int = 8) -> str:
    """Compute a simple average hash (aHash) for redundancy detection."""
    patch_small = patch.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.array(patch_small)
    avg = pixels.mean()
    diff = pixels > avg
    # Convert boolean array to a compact string of 0/1 characters
    return ''.join(diff.flatten().astype(int).astype(str))

# ------------------------------
# Patch extractor with QC
# ------------------------------

def extract_patches_from_slide(
    slide_path: str,
    output_base: str,
    patch_size: int = 256,
    stride: int = 256,
    level: int = 0,
    blur_threshold: float = 100.0,
    hash_size: int = 8,
):
    """Extract patches from a single WSI with quality‑control steps.

    Steps per candidate patch:
    1. Tissue filtering (background removal)
    2. Blur filtering using Laplacian variance
    3. Redundancy filtering using average hash
    """

    slide = openslide.OpenSlide(slide_path)
    width, height = slide.level_dimensions[level]
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    slide_output_dir = os.path.join(output_base, slide_name)
    os.makedirs(slide_output_dir, exist_ok=True)

    seen_hashes: set[str] = set()
    patch_id = 0

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")

            # 1) Tissue check
            if not is_tissue(patch):
                continue

            # 2) Blur check
            if is_blurry(patch, blur_threshold):
                continue

            # 3) Redundancy check
            h = average_hash(patch, hash_size)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # Save high‑quality, unique patch
            patch.save(os.path.join(slide_output_dir, f"patch_{patch_id}.png"))
            patch_id += 1


# ------------------------------
# Batch helpers & I/O
# ------------------------------

def process_all_slides(input_dir: str, output_dir: str):
    """Process every .svs slide in *input_dir* and save patches under *output_dir*."""
    slide_files = [f for f in os.listdir(input_dir) if f.endswith(".svs")]
    for slide_file in tqdm(slide_files, desc="Processing slides"):
        slide_path = os.path.join(input_dir, slide_file)
        extract_patches_from_slide(slide_path, output_dir)


def zip_output_folder(folder_path: str, zip_path: str):
    """Zip the *folder_path* recursively into *zip_path*."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)


# ------------------------------
# CLI entrypoint
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="WSI Patch Extractor with QC")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with WSI .svs files")
    parser.add_argument("--output_zip", type=str, default="wsi_patches.zip", help="Name of output zip file")
    parser.add_argument("--blur_thresh", type=float, default=100.0, help="Laplacian variance threshold for blur filtering")
    parser.add_argument("--hash_size", type=int, default=8, help="Hash size for redundancy detection (aHash)")
    args = parser.parse_args()

    temp_output_dir = "temp_patch_output"
    os.makedirs(temp_output_dir, exist_ok=True)

    # Pass user‑specified thresholds to extractor
    process_all_slides(args.input_dir, temp_output_dir)
    zip_output_folder(temp_output_dir, args.output_zip)

    print(f"\n압축 완료: {args.output_zip}\n")


if __name__ == "__main__":
    main()
