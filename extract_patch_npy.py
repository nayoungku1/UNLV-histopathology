import os
import openslide
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import argparse
import random

# ------------------------------
def is_tissue(patch, threshold: float = 0.8) -> bool:
    np_patch = np.array(patch.convert("RGB"))
    gray = np_patch.mean(axis=2)
    bg_ratio = (gray > 220).sum() / (gray.shape[0] * gray.shape[1])
    return bg_ratio < threshold


def is_blurry(patch, threshold: float = 100.0) -> bool:
    np_patch = np.array(patch.convert("L"))
    variance = cv2.Laplacian(np_patch, cv2.CV_64F).var()
    return variance < threshold


def extract_random_patches(slide, num_patches, patch_size, level):
    patches = []
    width, height = slide.level_dimensions[level]
    for _ in range(num_patches * 2):  # Try more than needed to account for filtering
        x = random.randint(0, width - patch_size)
        y = random.randint(0, height - patch_size)
        patch = slide.read_region((x * (2 ** level), y * (2 ** level)), level, (patch_size, patch_size)).convert("RGB")
        if is_tissue(patch) and not is_blurry(patch):
            patches.append(np.array(patch))
            if len(patches) >= num_patches:
                break
    return patches


def process_slide(slide_path, output_dir, num_patches, patch_size, level):
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    output_path = os.path.join(output_dir, f"{slide_name}.npz")
    try:
        slide = openslide.OpenSlide(slide_path)
        patches = extract_random_patches(slide, num_patches, patch_size, level)
        patches = np.stack(patches)
        np.savez_compressed(output_path, patches=patches)
        print(f"Saved {len(patches)} patches to {output_path}")
    except Exception as e:
        print(f"Failed to process {slide_path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input folder containing WSI files")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder")
    parser.add_argument("-c", type=str, choices=["Random"], default="Random", help="Patch selection mode")
    parser.add_argument("-n", type=int, default=2000, help="Number of patches (only used for -c Random)")
    parser.add_argument("--patch_size", type=int, default=224, help="Size of patches")
    parser.add_argument("--level", type=int, default=0, help="Level to extract patches from")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    svs_files = [f for f in os.listdir(args.input) if f.endswith(".svs")]

    for svs_file in tqdm(svs_files, desc="Processing slides"):
        slide_path = os.path.join(args.input, svs_file)
        process_slide(slide_path, args.output, args.n, args.patch_size, args.level)


if __name__ == "__main__":
    main()