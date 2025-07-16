import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import matplotlib.pyplot as plt

torch.manual_seed(314)

# Command-line argument parser
parser = argparse.ArgumentParser(description='Binary Classification Model Training')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--num_fc_layers', type=int, default=3, help='Number of fully connected layers')
args = parser.parse_args()

# 데이터셋 클래스 정의
class TileDataset(Dataset):
    def __init__(self, embeddings_path, labels_path):
        self.embeddings = torch.load(embeddings_path, weights_only=True)
        self.labels = torch.load(labels_path, weights_only=True)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# 경로 설정
train_embeddings_path = '/home/s25piteam/UNLV-histopathology/ViT/embeddings/train_tile_embeddings.pt'
train_labels_path = '/home/s25piteam/UNLV-histopathology/ViT/embeddings/train_tile_labels.pt'
val_embeddings_path = '/home/s25piteam/UNLV-histopathology/ViT/embeddings/val_tile_embeddings.pt'
val_labels_path = '/home/s25piteam/UNLV-histopathology/ViT/embeddings/val_tile_labels.pt'
test_embeddings_path = '/home/s25piteam/UNLV-histopathology/ViT/embeddings/test_tile_embeddings.pt'  
test_labels_path = '/home/s25piteam/UNLV-histopathology/ViT/embeddings/test_tile_labels.pt'  

# 데이터셋 로딩
train_dataset = TileDataset(train_embeddings_path, train_labels_path)
val_dataset = TileDataset(val_embeddings_path, val_labels_path)
test_dataset = TileDataset(test_embeddings_path, test_labels_path) 

# DataLoader 설정
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

# 모델 정의 (동적으로 Fully Connected Layer 생성)
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim, num_fc_layers):
        super(BinaryClassificationModel, self).__init__()
        
        # 동적으로 레이어 생성
        layers = []
        current_dim = input_dim
        hidden_dims = [2048, 1024, 512, 256, 128, 64]  # 기본 은닉층 크기
        num_layers = min(num_fc_layers, len(hidden_dims) + 1)  # 최대 레이어 수 제한
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            current_dim = hidden_dims[i]
        
        # 마지막 출력층
        layers.append(nn.Linear(current_dim, 1))
        
        # Sequential로 레이어 구성
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)  # 출력은 [0, 1] 사이로 설정
        return x

# 모델 초기화
input_dim = train_dataset.embeddings.shape[1]  # 임베딩 차원
model = BinaryClassificationModel(input_dim, args.num_fc_layers)

# 모델 구조 출력
print("Model Structure:")
print(model)

# 손실 함수 및 최적화 방법
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# GPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 함수
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        preds = (outputs > 0.5).float()
        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.size(0)
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

# 평가 함수
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            running_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
    
    avg_loss = running_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

# Test 함수
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            running_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
    
    avg_loss = running_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

# Learning curve plotting 함수
def plot_learning_curve(train_losses, val_losses, title, filename):
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# 학습 과정
train_losses = []
val_losses = []
for epoch in range(args.num_epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch [{epoch+1}/{args.num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Test set 평가
test_loss, test_accuracy = test(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Learning curve 저장
output_dir = "learning_curve"
os.makedirs(output_dir, exist_ok=True)
title = f"File: {os.path.basename(__file__)} | Layers: {args.num_fc_layers} | Epochs: {args.num_epochs}"
filename = os.path.join(output_dir, f"learning_curve_{args.num_fc_layers}_layers_{args.num_epochs}_epochs.png")
plot_learning_curve(train_losses, val_losses, title, filename)
