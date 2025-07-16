import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import matplotlib.pyplot as plt
import torchvision.models as models
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
import numpy as np

torch.manual_seed(314)

# 모델 클래스 정의 (기존과 동일)
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim, num_fc_layers):
        super(BinaryClassificationModel, self).__init__()
        layers = []
        current_dim = input_dim
        hidden_dims = [2048, 1024, 512, 256, 128, 64]
        num_layers = min(num_fc_layers, len(hidden_dims) + 1)
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            current_dim = hidden_dims[i]
        
        layers.append(nn.Linear(current_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x

class InceptionV3BinaryClassifier(nn.Module):
    def __init__(self, input_dim, num_fc_layers=3):
        super(InceptionV3BinaryClassifier, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        self.inception.fc = nn.Identity()
        
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

from torchvision.models import resnet50, ResNet50_Weights

class ResNetBinaryClassifier(nn.Module):
    def __init__(self, input_dim, num_fc_layers=3, local_weights_path=None):
        super(ResNetBinaryClassifier, self).__init__()
        # Try to load pre-trained ResNet50 with weights
        try:
            if local_weights_path and os.path.exists(local_weights_path):
                # Load weights from local file if provided
                self.resnet = resnet50(weights=None)
                self.resnet.load_state_dict(torch.load(local_weights_path, weights_only=True))
                print(f"Loaded ResNet50 weights from {local_weights_path}")
            else:
                # Load pre-trained weights from torchvision
                self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                print("Loaded pre-trained ResNet50 weights from torchvision")
        except Exception as e:
            print(f"Error loading pre-trained weights: {e}")
            print("Falling back to untrained ResNet50 model")
            self.resnet = resnet50(weights=None)  # Initialize without pre-trained weights

        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Dynamically create FC layers
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
        layers = []
        current_dim = input_dim
        hidden_dims = [512, 256, 128][:num_fc_layers-1]
        
        conv_layers = [
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        ]
        self.conv_block = nn.Sequential(*conv_layers)
        
        fc_layers = []
        current_dim = 128
        for i in range(min(num_fc_layers-1, len(hidden_dims))):
            fc_layers.append(nn.Linear(current_dim, hidden_dims[i]))
            fc_layers.append(nn.BatchNorm1d(hidden_dims[i]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.3))
            current_dim = hidden_dims[i]
        
        fc_layers.append(nn.Linear(current_dim, 1))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        x = x.squeeze(-1)
        x = self.fc_layers(x)
        x = torch.sigmoid(x)
        return x

class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, bidirectional=True):
        super(LSTMBinaryClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                           bidirectional=bidirectional, batch_first=True)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        x = self.fc(hidden)
        x = torch.sigmoid(x)
        return x

class RNNBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(RNNBinaryClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        rnn_out, hidden = self.rnn(x)
        x = self.fc(hidden[-1])
        x = torch.sigmoid(x)
        return x

MODEL_MAPPING = {
    'fc': BinaryClassificationModel,
    'inception': InceptionV3BinaryClassifier,
    'resnet': ResNetBinaryClassifier,
    'cnn': CNNBinaryClassifier,
    'lstm': LSTMBinaryClassifier,
    'rnn': RNNBinaryClassifier
}

parser = argparse.ArgumentParser(description='이진 분류 모델 학습')
parser.add_argument('--num_epochs', type=int, default=10, help='학습 에포크 수')
parser.add_argument('--num_fc_layers', type=int, default=3, help='완전 연결 레이어 수')
parser.add_argument('--model_type', type=str, default='fc',
                    choices=['fc', 'inception', 'resnet', 'cnn', 'lstm', 'rnn'],
                    help='Model Type: fc, inception, resnet, cnn, lstm, rnn')
args = parser.parse_args()

class TileDataset(Dataset):
    def __init__(self, embeddings_path, labels_path):
        self.embeddings = torch.load(embeddings_path, weights_only=True)
        self.labels = torch.load(labels_path, weights_only=True)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

train_embeddings_path = '/home/s25piteam/UNLV-histopathology/ViT/embeddings/train_tile_embeddings.pt'
train_labels_path = '/home/s25piteam/UNLV-histopathology/ViT/embeddings/train_tile_labels.pt'
val_embeddings_path = '/home/s25piteam/UNLV-histopathology/ViT/embeddings/val_tile_embeddings.pt'
val_labels_path = '/home/s25piteam/UNLV-histopathology/ViT/embeddings/val_tile_labels.pt'
test_embeddings_path = '/home/s25piteam/UNLV-histopathology/ViT/embeddings/test_tile_embeddings.pt'
test_labels_path = '/home/s25piteam/UNLV-histopathology/ViT/embeddings/test_tile_labels.pt'

train_dataset = TileDataset(train_embeddings_path, train_labels_path)
val_dataset = TileDataset(val_embeddings_path, val_labels_path)
test_dataset = TileDataset(test_embeddings_path, test_labels_path)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_dim = train_dataset.embeddings.shape[1]
model_class = MODEL_MAPPING[args.model_type]
if args.model_type in ['lstm', 'rnn']:
    model = model_class(input_dim=input_dim)
else:
    model = model_class(input_dim=input_dim, num_fc_layers=args.num_fc_layers)

print(f"선택된 모델: {args.model_type}")
print("모델 구조:")
print(model)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.close()

def plot_roc_curve(y_true, y_score, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()
    return roc_auc

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_outputs = []
    
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
        
        all_labels.extend(labels.cpu().numpy())
        all_outputs.extend(outputs.squeeze().detach().cpu().numpy())
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    auc = roc_auc_score(all_labels, all_outputs)
    
    return avg_loss, accuracy, auc, all_labels, all_outputs

def evaluate(model, loader, criterion, device, dataset_type='Validation'):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.squeeze().detach().cpu().numpy())
    
    avg_loss = running_loss / len(loader)
    accuracy = correct_predictions / total_predictions
    auc = roc_auc_score(all_labels, all_outputs)
    
    return avg_loss, accuracy, auc, all_labels, all_outputs

def plot_learning_curve(train_losses, val_losses, train_aucs, val_aucs, title, filename):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(train_aucs) + 1), train_aucs, label='Train AUC')
    plt.plot(range(1, len(val_aucs) + 1), val_aucs, label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC Curves')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 학습 과정
output_dir = "learning_curve"
os.makedirs(output_dir, exist_ok=True)

train_losses, val_losses = [], []
train_aucs, val_aucs = [], []

for epoch in range(args.num_epochs):
    train_loss, train_accuracy, train_auc, _, _ = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy, val_auc, _, _ = evaluate(model, val_loader, criterion, device, 'Validation')
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_aucs.append(train_auc)
    val_aucs.append(val_auc)
    
    print(f"Epoch [{epoch+1}/{args.num_epochs}], "
          f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Train AUC: {train_auc:.4f}, "
          f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f} | Validation AUC: {val_auc:.4f}")

# 최종 평가 및 시각화
# 학습 곡선 저장
source_file = os.path.basename(__file__)
title = (f"File: {source_file} | Model: {args.model_type} | "
         f"Layers: {args.num_fc_layers} | Epochs: {args.num_epochs}")
filename = os.path.join(output_dir, 
                       f"learning_curve_{source_file}_{args.model_type}_{args.num_fc_layers}_layers_{args.num_epochs}_epochs.png")
plot_learning_curve(train_losses, val_losses, train_aucs, val_aucs, title, filename)

# 최종 Train, Validation, Test 데이터셋 평가 및 시각화
datasets = {
    'train': train_loader,
    'validation': val_loader,
    'test': test_loader
}

for dataset_type, loader in datasets.items():
    loss, accuracy, auc, all_labels, all_outputs = evaluate(model, loader, criterion, device, dataset_type.capitalize())
    print(f"{dataset_type.capitalize()} Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    # Confusion Matrix
    plot_confusion_matrix(all_labels, (np.array(all_outputs) > 0.5).astype(int),
                         f'{dataset_type.capitalize()} Confusion Matrix',
                         os.path.join(output_dir, f'cm_{dataset_type}_{args.model_type}_{args.num_fc_layers}_layers_{args.num_epochs}_epochs.png'))
    
    # ROC Curve
    plot_roc_curve(all_labels, all_outputs,
                   f'{dataset_type.capitalize()} ROC Curve',
                   os.path.join(output_dir, f'roc_{dataset_type}_{args.model_type}_{args.num_fc_layers}_layers_{args.num_epochs}_epochs.png'))