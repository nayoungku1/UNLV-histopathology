#!/usr/bin/env python
# extract_patches.py

"""
Extract tissue patches from every WSI slide in CMB-LCA/.
Patches are stored per-sample in output/<slide_id>/.
Creates patch_index.csv mapping each patch to its slide-level label.
"""

import os
import openslide
import numpy as np
from PIL import Image
from tqdm import tqdm
import zipfile
import argparse

# 조직 포함 여부를 판단하는 함수
def is_tissue(patch, threshold=0.8):
    np_patch = np.array(patch.convert("RGB"))
    gray = np_patch.mean(axis=2)
    bg_ratio = (gray > 220).sum() / (gray.shape[0] * gray.shape[1])
    return bg_ratio < threshold

# 하나의 WSI 파일에서 패치 생성 및 저장
def extract_patches_from_slide(slide_path, output_base, patch_size=256, stride=256, level=0):
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.level_dimensions[level]
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    slide_output_dir = os.path.join(output_base, slide_name)
    os.makedirs(slide_output_dir, exist_ok=True)

    patch_id = 0
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = slide.read_region((x, y), level, (patch_size, patch_size))
            if is_tissue(patch):
                patch = patch.convert("RGB")
                patch.save(os.path.join(slide_output_dir, f"patch_{patch_id}.png"))
                patch_id += 1

# 전체 디렉토리에서 모든 WSI에 대해 패치 생성
def process_all_slides(input_dir, output_dir):
    slide_files = [f for f in os.listdir(input_dir) if f.endswith(".svs")]
    for slide_file in tqdm(slide_files, desc="Processing slides"):
        slide_path = os.path.join(input_dir, slide_file)
        extract_patches_from_slide(slide_path, output_dir)

# 결과를 압축
def zip_output_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

# 메인 함수
def main():
    parser = argparse.ArgumentParser(description="WSI Patch Extractor")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with WSI files')
    parser.add_argument('--output_zip', type=str, default='wsi_patches.zip', help='Name of output zip file')
    args = parser.parse_args()

    temp_output_dir = "temp_patch_output"
    os.makedirs(temp_output_dir, exist_ok=True)

    process_all_slides(args.input_dir, temp_output_dir)
    zip_output_folder(temp_output_dir, args.output_zip)

    print(f"압축 완료: {args.output_zip}")

if __name__ == "__main__":
    main()

