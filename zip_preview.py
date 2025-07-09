#!/usr/bin/env python3
# zip_preview.py

import os
import argparse
import zipfile
import tempfile
import shutil
import matplotlib.pyplot as plt
from PIL import Image, PngImagePlugin

# PNG 텍스트 청크 제한 늘리기
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100MB

def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

def plot_and_save(img_path):
    folder_name = os.path.basename(os.path.normpath(img_path))
    img_files = [f for f in os.listdir(img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_imgs = len(img_files)

    if total_imgs == 0:
        print(f"⚠️  No images found in {img_path}")
        return

    n_imgs = min(20, total_imgs)
    dir_size_bytes = get_dir_size(img_path)
    dir_size_mb = dir_size_bytes / (1024 * 1024)

    cols = 5
    rows = (n_imgs + cols - 1) // cols

    plt.figure(figsize=(15, 3 * rows))

    for i in range(n_imgs):
        img_file = img_files[i]
        img = Image.open(os.path.join(img_path, img_file))

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{img_file}')

    plt.suptitle(
        f'Folder: {folder_name} | Size: {dir_size_mb:.2f} MB | Format: PNG | Total Images: {total_imgs}',
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_file = f'{folder_name}_preview.png'
    plt.savefig(output_file, dpi=300)
    print(f'✅ Saved preview image to: {output_file}')

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Unzip and plot images in each folder inside a ZIP file")
    parser.add_argument("zipfile", type=str, help="Path to the ZIP file")
    args = parser.parse_args()

    zip_path = args.zipfile

    if not os.path.isfile(zip_path):
        print(f"❌ File not found: {zip_path}")
        return

    # 임시 디렉토리 생성 후 압축 해제
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📂 Extracting {zip_path} to {temp_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # 최상위 폴더 찾기
        top_level_dirs = [os.path.join(temp_dir, d) for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]

        if not top_level_dirs:
            print("❌ No folders found inside the ZIP.")
            return

        for folder in top_level_dirs:
            print(f"🔍 Processing folder: {folder}")
            plot_and_save(folder)

if __name__ == "__main__":
    main()
