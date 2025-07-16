import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import matplotlib.pyplot as plt
import torchvision.models as models

torch.manual_seed(314)

# 모델 클래스 정의
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim, num_fc_layers):
        super(BinaryClassificationModel, self).__init__()
        # 동적으로 FC 레이어 생성
        layers = []
        current_dim = input_dim
        hidden_dims = [2048, 1024, 512, 256, 128, 64]  # 기본 은닉층 크기
        num_layers = min(num_fc_layers, len(hidden_dims) + 1)  # 최대 레이어 수 제한
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            current_dim = hidden_dims[i]
        
        # 마지막 출력층
        layers.append(nn.Linear(current_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)  # 출력 [0, 1] 범위로 설정
        return x

class InceptionV3BinaryClassifier(nn.Module):
    def __init__(self, input_dim, num_fc_layers=3):
        super(InceptionV3BinaryClassifier, self).__init__()
        # 사전 학습된 InceptionV3 로드 및 수정
        self.inception = models.inception_v3(pretrained=True)
        self.inception.fc = nn.Identity()  # 기존 FC 레이어 제거
        
        # 동적으로 FC 레이어 생성
        layers = []
        current_dim = input_dim
        hidden_dims = [512, 256, 128][:num_fc_layers-1]
        for i in range(min(num_fc_layers-1, len(hidden_dims))):
            layers.append(nn.Linear(current_dim, hidden_dims[i]))
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            current_dim = hidden_dims[i]
        
        layers.append(nn.Linear(current_dim, 1))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)
        x = torch.sigmoid(x)
        return x

class ResNetBinaryClassifier(nn.Module):
    def __init__(self, input_dim, num_fc_layers=3):
        super(ResNetBinaryClassifier, self).__init__()
        # 사전 학습된 ResNet50 로드 및 수정
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        
        # 동적으로 FC 레이어 생성
        layers = []
        current_dim = input_dim
        hidden_dims = [512, 256, 128][:num_fc_layers-1]
        for i in range(min(num_fc_layers-1, len(hidden_dims))):
            layers.append(nn.Linear(current_dim, hidden_dims[i]))
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            current_dim = hidden_dims[i]
        
        layers.append(nn.Linear(current_dim, 1))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)
        x = torch.sigmoid(x)
        return x

class CNNBinaryClassifier(nn.Module):
    def __init__(self, input_dim, num_fc_layers=3):
        super(CNNBinaryClassifier, self).__init__()
        # 1D CNN 레이어 정의
        layers = []
        current_dim = input_dim
        hidden_dims = [512, 256, 128][:num_fc_layers-1]
        
        conv_layers = [
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  # Conv1d 후 배치 정규화
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),  # Conv1d 후 배치 정규화
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        ]
        self.conv_block = nn.Sequential(*conv_layers)
        
        # FC 레이어 정의
        fc_layers = []
        current_dim = 128
        for i in range(min(num_fc_layers-1, len(hidden_dims))):
            fc_layers.append(nn.Linear(current_dim, hidden_dims[i]))
            fc_layers.append(nn.BatchNorm1d(hidden_dims[i]))  # Linear 후 배치 정규화
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.3))
            current_dim = hidden_dims[i]
        
        fc_layers.append(nn.Linear(current_dim, 1))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        x = self.conv_block(x)  # [batch_size, 128, 1]
        x = x.squeeze(-1)  # [batch_size, 128]
        x = self.fc_layers(x)
        x = torch.sigmoid(x)
        return x

class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, bidirectional=True):
        super(LSTMBinaryClassifier, self).__init__()
        # LSTM 레이어 정의
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                           bidirectional=bidirectional, batch_first=True)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        # FC 레이어 정의
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim  # 양방향 LSTM의 경우 256
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64),  # 입력 차원 수정: 256 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)  # 입력 차원 수정: 16 -> 1
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        lstm_out, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch_size, 256]
        else:
            hidden = hidden[-1]  # [batch_size, 128]
        x = self.fc(hidden)
        x = torch.sigmoid(x)
        return x

class RNNBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(RNNBinaryClassifier, self).__init__()
        # RNN 레이어 정의
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
        
        # FC 레이어 정의
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),  # 128 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),  # 64 -> 16
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)  # 입력 차원 수정: 32 -> 16
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        rnn_out, hidden = self.rnn(x)
        x = self.fc(hidden[-1])  # hidden[-1]: [batch_size, hidden_dim]
        x = torch.sigmoid(x)
        return x

# 지원하는 모델 매핑
MODEL_MAPPING = {
    'fc': BinaryClassificationModel,
    'inception': InceptionV3BinaryClassifier,
    'resnet': ResNetBinaryClassifier,
    'cnn': CNNBinaryClassifier,
    'lstm': LSTMBinaryClassifier,
    'rnn': RNNBinaryClassifier
}

# Command-line argument parser
parser = argparse.ArgumentParser(description='이진 분류 모델 학습')
parser.add_argument('--num_epochs', type=int, default=10, help='학습 에포크 수')
parser.add_argument('--num_fc_layers', type=int, default=3, help='완전 연결 레이어 수')
parser.add_argument('--model_type', type=str, default='fc',
                    choices=['fc', 'inception', 'resnet', 'cnn', 'lstm', 'rnn'],
                    help='Model Type: fc, inception, resnet, cnn, lstm, rnn')
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

# 모델 초기화
input_dim = train_dataset.embeddings.shape[1]  # 임베딩 차원
model_class = MODEL_MAPPING[args.model_type]
if args.model_type in ['lstm', 'rnn']:
    # LSTM 및 RNN은 num_fc_layers 대신 고정된 FC 레이어를 사용
    model = model_class(input_dim=input_dim)
else:
    model = model_class(input_dim=input_dim, num_fc_layers=args.num_fc_layers)

# 모델 구조 출력
print(f"선택된 모델: {args.model_type}")
print("모델 구조:")
print(model)

# 손실 함수 및 최적화 방법
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

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

# 테스트 함수
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

# 학습 곡선 플롯 함수
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
          f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}, "
          f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

# 테스트 데이터셋 평가
test_loss, test_accuracy = test(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 학습 곡선 저장
output_dir = "learning_curve"
os.makedirs(output_dir, exist_ok=True)
title = (f"File: {os.path.basename(__file__)} | Model: {args.model_type} | "
         f"Layers: {args.num_fc_layers} | Epochs: {args.num_epochs}")
filename = os.path.join(output_dir, 
                       f"learning_curve_{args.model_type}_{args.num_fc_layers}_layers_{args.num_epochs}_epochs.png")
plot_learning_curve(train_losses, val_losses, title, filename)