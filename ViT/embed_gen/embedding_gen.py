import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import timm

# ======================
# 설정 및 환경 초기화
# ======================
torch.manual_seed(314)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
tile_encoder = tile_encoder.to(device).eval()

label_csv_path = os.path.expanduser("~/UNLV-histopathology/ViT/embed_gen/label.csv")
local_cache_dir = os.path.expanduser("~/UNLV-histopathology/ViT/embed_gen/npz_cache")
os.makedirs(local_cache_dir, exist_ok=True)

# ======================
# Transform 정의
# ======================
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# ======================
# NPZ 로딩 함수 (로컬 전용)
# ======================
def load_npz_local_only(filename):
    local_path = os.path.join(local_cache_dir, filename)
    if os.path.exists(local_path):
        try:
            return np.load(local_path)
        except Exception as e:
            print(f"⚠️ 로컬 파일 손상됨: {filename} — {e}")
    else:
        print(f"❌ 로컬에 없음: {filename}")
    return None

# ======================
# 타겟 파일 리스트
# ======================
target_npz_files = [
    'MSB-07219-02-01.npz', 'MSB-08777-01-02.npz', 'MSB-05837-01-06.npz',
    'MSB-04286-01-01.npz', 'MSB-05756-01-02.npz', 'MSB-00179-03-20.npz',
    'MSB-01772-06-02.npz', 'MSB-06824-08-06.npz', 'MSB-06150-02-05.npz',
    'MSB-02498-01-04.npz', 'MSB-01433-03-02.npz', 'MSB-05767-07-06.npz',
    'MSB-04567-03-20.npz', 'MSB-05388-01-02.npz', 'MSB-04315-01-02.npz',
    'MSB-04248-03-01.npz', 'MSB-03410-01-02.npz', 'MSB-09117-03-06.npz',
    'MSB-09666-02-02.npz', 'MSB-02151-02-14.npz', 'MSB-06150-02-04.npz',
    'MSB-09466-02-02.npz', 'MSB-02428-03-02.npz', 'MSB-08928-01-06.npz',
    'MSB-06150-02-12.npz', 'MSB-07572-01-02.npz', 'MSB-02151-01-20.npz',
    'MSB-05876-01-07.npz', 'MSB-08242-01-05.npz', 'MSB-09505-01-02.npz',
    'MSB-08063-04-01.npz'
]

# ======================
# 누락된 파일 확인
# ======================
missing = []
for fname in target_npz_files:
    if not os.path.exists(os.path.join(local_cache_dir, fname)):
        missing.append(fname)

if missing:
    print("❌ 누락된 파일이 있습니다. 다음 파일을 npz_cache 폴더에 복사하세요:")
    for m in missing:
        print("-", m)
else:
    print("✅ 모든 파일이 존재합니다.")

# ======================
# 메타데이터 필터링
# ======================
df = pd.read_csv(label_csv_path)
df['pub_subspec_id'] = df['pub_subspec_id'].apply(lambda x: x if x.endswith('.npz') else f"{x}.npz")
filtered_df = df[df['pub_subspec_id'].isin(target_npz_files)].reset_index(drop=True)

# ======================
# Data Index 생성
# ======================
def make_stratified_data_index_local(filtered_df):
    filename_to_label = dict(zip(filtered_df['pub_subspec_id'], filtered_df['label']))
    data_index = []

    for fname, label in filename_to_label.items():
        npz = load_npz_local_only(fname)
        if npz is None:
            continue

        for key in npz.files:
            patch_array = npz[key]
            if patch_array.ndim == 4:
                for i in range(patch_array.shape[0]):
                    data_index.append((fname, key, i, label))
            else:
                data_index.append((fname, key, None, label))
    return data_index

data_index = make_stratified_data_index_local(filtered_df)

# ======================
# Stratified Split
# ======================
def stratified_split(data_index, train_ratio=0.7, val_ratio=0.15, seed=314):
    label_to_items = defaultdict(list)
    for item in data_index:
        label = item[3]
        label_to_items[label].append(item)

    train, val, test = [], [], []
    random.seed(seed)

    for label, items in label_to_items.items():
        random.shuffle(items)
        n_total = len(items)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    return train, val, test

train_index, val_index, test_index = stratified_split(data_index)

# ======================
# Dataset 정의
# ======================
class PatchDataset(Dataset):
    def __init__(self, data_index, transform=None):
        self.data_index = data_index
        self.transform = transform

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        fname, key, patch_idx, label = self.data_index[idx]
        npz = load_npz_local_only(fname)
        patch_array = npz[key]

        patch = patch_array[patch_idx] if patch_idx is not None else patch_array
        patch = Image.fromarray(patch.astype(np.uint8))

        if self.transform:
            patch = self.transform(patch)

        return patch, int(label)

# ======================
# DataLoader 정의
# ======================
train_loader = DataLoader(PatchDataset(train_index, transform), batch_size=32, shuffle=False, num_workers=2)
val_loader = DataLoader(PatchDataset(val_index, transform), batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(PatchDataset(test_index, transform), batch_size=32, shuffle=False, num_workers=2)

# ======================
# 임베딩 생성 함수
# ======================
def generate_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_embeddings), torch.cat(all_labels)

# ======================
# 임베딩 저장
# ======================
train_embeddings, train_labels = generate_embeddings(tile_encoder, train_loader, device)
torch.save(train_embeddings, "train_tile_embeddings.pt")
torch.save(train_labels, "train_tile_labels.pt")

val_embeddings, val_labels = generate_embeddings(tile_encoder, val_loader, device)
torch.save(val_embeddings, "val_tile_embeddings.pt")
torch.save(val_labels, "val_tile_labels.pt")

test_embeddings, test_labels = generate_embeddings(tile_encoder, test_loader, device)
torch.save(test_embeddings, "test_tile_embeddings.pt")
torch.save(test_labels, "test_tile_labels.pt")
