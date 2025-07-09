#!/usr/bin/env python3
# save_preview.py

import os
import argparse
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

def main():
    # Parser 설정
    parser = argparse.ArgumentParser(description="Plot sample images from a directory and save as PNG")
    parser.add_argument("--dir", type=str, help="Path to the image directory")
    args = parser.parse_args()

    img_path = args.dir

    # 폴더명
    folder_name = os.path.basename(os.path.normpath(img_path))

    # 이미지 파일 리스트
    img_files = [f for f in os.listdir(img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_imgs = len(img_files)

    if total_imgs == 0:
        print(f"No images found in {img_path}")
        return

    n_imgs = min(20, total_imgs)

    # 폴더 용량 (MB)
    dir_size_bytes = get_dir_size(img_path)
    dir_size_mb = dir_size_bytes / (1024 * 1024)

    # subplot
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

    # 결과 PNG 저장
    output_file = f'{folder_name}_preview.png'
    plt.savefig(output_file, dpi=300)
    print(f'Saved preview image to: {output_file}')

    plt.show()

if __name__ == "__main__":
    main()
