import os
import random
import numpy as np
import librosa
import librosa.display
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


def extract_audio_features(y, sr=16000, duration=3):
    max_len = sr * duration
    # Fix length: truncate or pad the audio to match the target length
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]
    
    # ✅ Normalize waveform to avoid underflow
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    # rmse = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Normalize each feature
    def norm(x): return librosa.util.normalize(x, axis=1)

    feature_stack = np.vstack([
        norm(mfcc),
        norm(delta),
        norm(delta2),
        # norm(chroma),
        norm(contrast),
        norm(zcr),
        # norm(rmse),
        norm(centroid)
    ])  # Shape: (features, frames)

    return feature_stack.T  # Shape: (frames, features)

class RevDeEsDataset(Dataset):
    def __init__(self, file_paths, labels=None, sr=16000, duration=3):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.max_len = sr * duration

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        y, _ = librosa.load(path, sr=self.sr)
        # Fix the length of the audio
        if len(y) < self.max_len:
            y = np.pad(y, (0, self.max_len - len(y)))  # Pad with zeros if shorter
        else:
            y = y[:self.max_len]  # Truncate if longer
    
        features = extract_audio_features(y, sr=self.sr)  # Extract audio features (e.g., MFCC, delta, etc.)
        features = torch.tensor(features, dtype=torch.float32)
    
        if self.labels is not None:
            label = self.labels[idx]
            return features, label
        else:
            return features, path

    def __len__(self):
        return len(self.file_paths)


import os
from glob import glob

def get_audio_filepaths_and_labels(root_dir, exts=[".wav", ".mp3"]):
    file_paths = []
    labels = []
    class_to_idx = {}

    for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        class_to_idx[class_name] = idx
        for ext in exts:
            files = glob(os.path.join(class_path, f"*{ext}"))
            file_paths.extend(files)
            labels.extend([idx] * len(files))

    return file_paths, labels, class_to_idx

train_dir = "/kaggle/working/audio/train"  # <- This contains subfolders like angry/, happy/, etc.
valid_dir = "/kaggle/working/audio/valid"  # <- This contains subfolders like angry/, happy/, etc.

train_paths, train_labels, label_map = get_audio_filepaths_and_labels(train_dir)
val_file_paths, val_labels, label_map = get_audio_filepaths_and_labels(valid_dir)

print(f"Total files: {len(train_paths)}")
print(f"Label Mapping: {label_map}")


train_set = RevDeEsDataset(train_paths, train_labels)
val_set = RevDeEsDataset(val_file_paths, val_labels)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
# Get feature dimensions for model input
sample_x, _ = train_set[0]
features_time_steps, features_dim = sample_x.shape
sample_x.shape

NUM_CLASSES = len(set(train_labels))
NUM_CLASSES, len(train_loader), len(val_loader)




##########################
#Use Attention After RNN : Add attention after the LSTM output to weigh different time steps.
#########################
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        weights = torch.softmax(self.attn(x), dim=1)
        context = (x * weights).sum(dim=1)
        return context


# ===========================
# CRNN Model
# ===========================
class CRNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d((2, 2)),
        
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        cnn_out_time = input_shape[0] // 8 #CNN has 3 maxpool layers, each halving both the time and frequency axes, so both T and F are divided by 8
        cnn_out_freq = input_shape[1] // 8
        # self.rnn = nn.LSTM(input_size=cnn_out_freq * 64, hidden_size=128, batch_first=True, bidirectional=True)
        # self.fc = nn.Linear(128 * 2, num_classes)
        self.rnn = nn.LSTM(input_size=cnn_out_freq * 128, hidden_size=256,
                   batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)

        self.attention = Attention(512)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.cnn(x)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x, _ = self.rnn(x)
        # x = self.fc(x[:, -1])
        x = self.attention(x)
        x = self.fc(x)
        return x


# Check for GPU (CUDA) availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CRNN(input_shape=(features_time_steps, features_dim), num_classes=NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ===========================
# Train / Validate
# ===========================
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        x = x.unsqueeze(1)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)  # Adjust this as per model
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()

            probs = F.softmax(out, dim=1)  # Get predicted probabilities
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return total_loss / len(dataloader), acc, f1, all_preds, all_labels, all_probs


train_losses, val_losses, val_accuracies, val_f1s = [], [], [], []


for epoch in range(50):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, val_f1, val_preds, val_labels, val_probs = validate(model, val_loader, criterion, device, num_classes=NUM_CLASSES)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_f1s.append(val_f1)
    print(f"Epoch {epoch+1}: TL {train_loss:.4f}, VL {val_loss:.4f}, Acc {val_acc:.4f}, F1 {val_f1:.4f}")




## Evaluations