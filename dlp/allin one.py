import os, shutil, pathlib, cv2
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import librosa, librosa.display
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import Conv2d
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import entropy
from glob import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from moviepy.editor import VideoFileClip
import kagglehub
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

# Feature extraction
def extract_audio_features(y, sr=16000, duration=3):
    y = np.pad(y, (0, sr*duration - len(y))) if len(y) < sr*duration else y[:sr*duration]
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    features = [librosa.util.normalize(librosa.feature.mfcc(y, sr=sr, n_mfcc=40)), 
                librosa.util.normalize(librosa.feature.delta(librosa.feature.mfcc(y, sr=sr, n_mfcc=40))),
                librosa.util.normalize(librosa.feature.delta(librosa.feature.mfcc(y, sr=sr, n_mfcc=40), order=2)),
                librosa.util.normalize(librosa.feature.spectral_contrast(y, sr=sr)),
                librosa.util.normalize(librosa.feature.zero_crossing_rate(y)),
                librosa.util.normalize(librosa.feature.spectral_centroid(y, sr=sr))]
    return np.vstack(features).T

# Dataset class
class RevDeEsDataset(Dataset):
    def __init__(self, file_paths, labels=None, sr=16000, duration=3):
        self.file_paths, self.labels, self.sr, self.max_len = file_paths, labels, sr, sr * duration

    def __getitem__(self, idx):
        y, _ = librosa.load(self.file_paths[idx], sr=self.sr)
        y = np.pad(y, (0, self.max_len - len(y))) if len(y) < self.max_len else y[:self.max_len]
        features = torch.tensor(extract_audio_features(y, sr=self.sr), dtype=torch.float32)
        return (features, self.labels[idx]) if self.labels is not None else (features, self.file_paths[idx])

    def __len__(self): return len(self.file_paths)

# Get file paths and labels
def get_audio_filepaths_and_labels(root_dir, exts=[".wav", ".mp3"]):
    file_paths, labels, class_to_idx = [], [], {}
    for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            class_to_idx[class_name] = idx
            for ext in exts:
                files = glob(os.path.join(class_path, f"*{ext}"))
                file_paths.extend(files)
                labels.extend([idx] * len(files))
    return file_paths, labels, class_to_idx

# Model classes
class Attention(nn.Module):
    def __init__(self, hidden_dim): super().__init__(); self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, x): return (x * torch.softmax(self.attn(x), dim=1)).sum(dim=1)

class CRNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) if i % 2 == 0 else nn.BatchNorm2d(out_channels) if i % 2 == 1 else nn.ReLU() if i % 2 == 2 else nn.MaxPool2d((2, 2)) for in_channels, out_channels in [(1, 32), (32, 64), (64, 128)] for i in range(4)])
        cnn_out_time, cnn_out_freq = input_shape[0] // 8, input_shape[1] // 8
        self.rnn = nn.LSTM(cnn_out_freq * 128, 256, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.attention = Attention(512)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x): x = self.cnn(x).permute(0, 2, 1, 3).contiguous().view(x.size(0), -1, x.size(1) * x.size(3)); x, _ = self.rnn(x); return self.fc(self.attention(x))

# Initialize
train_dir, valid_dir = "/kaggle/working/audio/train", "/kaggle/working/audio/valid"
train_paths, train_labels, label_map = get_audio_filepaths_and_labels(train_dir)
val_file_paths, val_labels, _ = get_audio_filepaths_and_labels(valid_dir)

train_set, val_set = RevDeEsDataset(train_paths, train_labels), RevDeEsDataset(val_file_paths, val_labels)
train_loader, val_loader = DataLoader(train_set, batch_size=32, shuffle=True), DataLoader(val_set, batch_size=32)
sample_x, _ = train_set[0]
NUM_CLASSES = len(set(train_labels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(input_shape=sample_x.shape, num_classes=NUM_CLASSES).to(device)
optimizer, criterion = torch.optim.Adam(model.parameters(), lr=1e-3), nn.CrossEntropyLoss()

# Training & Validation
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = sum(criterion(model(x.unsqueeze(1).to(device)), y.to(device)).backward() or loss.item() for x, y in dataloader)
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x.unsqueeze(1).to(device))
            loss = criterion(out, y.to(device))
            preds.extend(torch.argmax(F.softmax(out, dim=1), dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(F.softmax(out, dim=1).cpu().numpy())
    return loss.item(), accuracy_score(labels, preds), f1_score(labels, preds, average='weighted')

# Loop
train_losses, val_losses, val_accuracies, val_f1s = [], [], [], []
for epoch in range(50):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_f1s.append(val_f1)
    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Acc {val_acc:.4f}, F1 {val_f1:.4f}")



def extract_frames_and_audio_from_paths(video_paths, output_folder, audio_folder, frame_rate=1):
    output_folder, audio_folder = pathlib.Path(output_folder), pathlib.Path(audio_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    audio_folder.mkdir(parents=True, exist_ok=True)

    for idx, video_path in enumerate(video_paths):
        print(f"Processing video {idx + 1}: {video_path}")
        if not pathlib.Path(video_path).exists(): continue
        cap = cv2.VideoCapture(str(video_path)) if cap.isOpened() else None
        if not cap: continue
        video_name, count, saved = pathlib.Path(video_path).stem, 0, 0
        while (ret := cap.read())[0]:
            if count % frame_rate == 0: cv2.imwrite(str(output_folder / f"{video_name}_frame_{saved:04d}.jpg"), ret[1]); saved += 1
            count += 1
        cap.release()
        print(f"Extracted {saved} frames from {video_path.name}")
        try:
            clip = VideoFileClip(str(video_path))
            audio_path = audio_folder / f"{video_name}_audio.mp3"
            clip.audio.write_audiofile(str(audio_path)); save_spectrogram(audio_path)
        except Exception as e: print(f"Failed to extract audio from {video_path.name}: {e}")

def save_spectrogram(audio_path, sr=22050, n_mels=128):
    y, _ = librosa.load(audio_path, sr=sr)
    S_DB = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels), ref=np.max)
    plt.figure(figsize=(3, 3)); librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel'); plt.axis('off')
    plt.tight_layout(); plt.savefig(audio_path.with_suffix('.png'), bbox_inches='tight', pad_inches=0); plt.close()

# Data Preprocessing and Setup
extract_frames_and_audio_from_paths(final_df, 15)
train_data_dir, val_data_dir = pathlib.Path('/kaggle/working/frames/train'), pathlib.Path('/kaggle/working/frames/valid')

img_height, img_width, batch_size, AUTOTUNE = 224, 224, 32, tf.data.AUTOTUNE
train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_data_dir, seed=123, image_size=(img_height, img_width), batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_data_dir, seed=123, image_size=(img_height, img_width), batch_size=batch_size)
class_names = train_ds.class_names
print(f"Class Names: {class_names}")

# Preprocessing & Data Augmentation
train_ds, val_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, len(class_names)))).shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE), val_ds.map(lambda x, y: (x, tf.one_hot(y, len(class_names)))).cache().prefetch(buffer_size=AUTOTUNE)
data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomZoom(0.1), layers.RandomContrast(0.1)])

# Model Definition
resnet_model = Sequential([
    layers.InputLayer(input_shape=(img_height, img_width, 3)), data_augmentation,
    tf.keras.applications.ResNet50(include_top=False, input_shape=(img_height, img_width, 3), pooling='avg', weights='imagenet'),
    layers.Dense(512, activation='relu'), layers.Dropout(0.3), layers.Dense(len(class_names), activation='softmax')
])
resnet_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks & Training
checkpoint_callback = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[checkpoint_callback, early_stopping_callback])

# F1 Score Metric
from keras import backend as K
def f1_score(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    precision, recall = tp / (tp + fp + K.epsilon()), tp / (tp + fn + K.epsilon())
    return 2 * precision * recall / (precision + recall + K.epsilon())

# Evaluation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
y_true, y_pred = [], [np.argmax(resnet_model.predict(images), axis=1) for images, labels in val_ds]
y_true.extend(np.argmax(labels.numpy(), axis=1) for images, labels in val_ds)
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix"); plt.show()
print(classification_report(y_true, y_pred))



def save_spectrogram(audio_path, save_path, sr=22050, n_mels=128):
    y, _ = librosa.load(audio_path, sr=sr)
    S_DB = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels), ref=np.max)
    plt.figure(figsize=(3, 3)); librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off'); plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight', pad_inches=0); plt.close()

def convert_to_spectrograms(input_dir, output_dir): 
    for emotion in os.listdir(input_dir): 
        out_emotion_path = os.path.join(output_dir, emotion); os.makedirs(out_emotion_path, exist_ok=True)
        [save_spectrogram(os.path.join(input_dir, emotion, f), os.path.join(out_emotion_path, f.replace('.wav', '.png'))) for f in os.listdir(os.path.join(input_dir, emotion)) if f.endswith(".wav")]

def convert_dataframe_to_spectrograms(df, base_output_dir):
    [save_spectrogram(row['audio_paths'], os.path.join(base_output_dir, row['category'], str(row['emotion_label']), os.path.basename(row['audio_paths']).replace('.wav', '.png'))) for _, row in df.iterrows()]

shutil.rmtree('/kaggle/working/audio', ignore_errors=True)
final_df = pd.read_csv("mapping.csv")
base_output_dir = 'spectrograms'
convert_dataframe_to_spectrograms(final_df, base_output_dir)
convert_to_spectrograms("/kaggle/input/frames-audio-train-test-validate-revdess-dataset/audio/train", "spectrogramss_train")
convert_to_spectrograms("/kaggle/input/frames-audio-train-test-validate-revdess-dataset/audio/valid", "spectrogramss_valid")
convert_to_spectrograms("/kaggle/input/frames-audio-train-test-validate-revdess-dataset/audio/test", "spectrogramss_test")

transform_train = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)), transforms.ToTensor()])
transform_val = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_loader = DataLoader(datasets.ImageFolder('/kaggle/working/spectrograms/train', transform=transform_train), batch_size=32, shuffle=True)
val_loader = DataLoader(datasets.ImageFolder('/kaggle/working/spectrograms/valid', transform=transform_val), batch_size=32)
test_loader = DataLoader(datasets.ImageFolder('/kaggle/working/spectrograms/test', transform=transform_val), batch_size=32)

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=8): 
        super().__init__(); self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x): return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpectrogramCNN(num_classes=len(train_loader.dataset.classes)).to(device)
criterion, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    train_loss, train_preds, train_labels = 0, [], []
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs); loss = criterion(outputs, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1); train_preds.extend(preds.cpu().numpy()); train_labels.extend(labels.cpu().numpy())

    train_acc, train_f1 = accuracy_score(train_labels, train_preds), f1_score(train_labels, train_preds, average='weighted')

    model.eval()
    val_loss, val_preds, val_labels = 0, [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs); loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1); val_preds.extend(preds.cpu().numpy()); val_labels.extend(labels.cpu().numpy())

    val_acc, val_f1 = accuracy_score(val_labels, val_preds), f1_score(val_labels, val_preds, average='weighted')
    print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

def evaluate_model(model, dataloader, class_names):
    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images); preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    print(f"Classification Report:\n{classification_report(all_labels, all_preds, target_names=class_names)}")

evaluate_model(model, test_loader, class_names=val_loader.dataset.classes)

model.load_state_dict(torch.load("best_cnn_model.pth"))
model.eval()
target_layer = next(name for name, module in reversed(list(model.named_modules())) if isinstance(module, Conv2d))
cam_extractor = GradCAM(model, target_layer=target_layer)

img = Image.open("/kaggle/working/spectrogramss_test/angry/01-01-05-01-01-01-14_audio.png").convert("RGB")
input_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])(img).unsqueeze(0).to(device)
out = model(input_tensor)
class_id = out.argmax().item()
activation_map = cam_extractor(class_id, out)
heatmap = activation_map[0].squeeze().cpu().numpy()
result = overlay_mask(img, Image.fromarray(np.uint8(heatmap * 255), mode='L'), alpha=0.5)

plt.figure(figsize=(8, 8)); plt.imshow(result); plt.title(f"Grad-CAM for class {class_id}"); plt.axis("off"); plt.show()



adrivg_ravdess_emotional_speech_video_path = kagglehub.dataset_download('adrivg/ravdess-emotional-speech-video')
print('Data source import complete.')

def extract_audio_features(y, sr=16000, duration=3):
    max_len = sr * duration
    y = np.pad(y, (0, max_len - len(y))) if len(y) < max_len else y[:max_len]
    y /= np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1
    features = np.vstack([
        librosa.util.normalize(x, axis=1) for x in [librosa.feature.mfcc(y, sr), librosa.feature.delta(librosa.feature.mfcc(y, sr)),
                                                    librosa.feature.delta(librosa.feature.mfcc(y, sr), order=2),
                                                    librosa.feature.spectral_contrast(y, sr),
                                                    librosa.feature.zero_crossing_rate(y), librosa.feature.spectral_centroid(y, sr)]])
    return features.T

class RevDeEsDataset(Dataset):
    def __init__(self, file_paths, labels=None, sr=16000, duration=3):
        self.file_paths, self.labels, self.sr, self.duration = file_paths, labels, sr, duration
        self.max_len = sr * duration
    def __getitem__(self, idx):
        y, _ = librosa.load(self.file_paths[idx], sr=self.sr)
        y = np.pad(y, (0, self.max_len - len(y))) if len(y) < self.max_len else y[:self.max_len]
        features = extract_audio_features(y, sr=self.sr)
        return torch.tensor(features, dtype=torch.float32), self.labels[idx] if self.labels else (features, self.file_paths[idx])
    def __len__(self): return len(self.file_paths)

class Attention(nn.Module):
    def __init__(self, hidden_dim): super().__init__(); self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, x): return (x * torch.softmax(self.attn(x), dim=1)).sum(dim=1)

class CRNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(0.3), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 2)),
        )
        cnn_out_time, cnn_out_freq = input_shape[0] // 8, input_shape[1] // 8
        self.rnn = nn.LSTM(input_size=cnn_out_freq * 128, hidden_size=256, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.attention, self.fc = Attention(512), nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[2], -1)
        x, _ = self.rnn(x)
        return self.fc(self.attention(x))

def validate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = sum([criterion(model(x.unsqueeze(1).to(device)), y.to(device)).item() for x, y in dataloader])
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='weighted')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
audio_model = CRNN(input_shape=(94, 129), num_classes=8).to(device)
audio_model.load_state_dict(torch.load('/content/crnn_attention_final_model.pth')['model_state_dict'])

frame_model, spectrogram_model = load_model("/content/best_model(1).keras"), SpectrogramCNN(num_classes=8).to(device)
spectrogram_model.load_state_dict(torch.load("best_spectrogram_cnn.pth", map_location=device))
spectrogram_model.eval()

def one_hot_encode(batch_images, batch_labels): return batch_images, tf.one_hot(batch_labels, depth=8)

def safe_entropy(probs, eps=1e-6): return entropy(np.array(probs) / (np.sum(probs) + eps) + eps)

def predict_from_video(video_path, frame_dir, audio_path, frame_model, audio_model, device, video_acc=0.9146, audio_acc=0.9236):
    extract_frames_and_audio_from_paths([video_path], f"{frame_dir}/test", os.path.dirname(audio_path), frame_rate=30)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(pathlib.Path(frame_dir), seed=123, image_size=(224, 224), batch_size=32)
    test_ds = test_ds.map(one_hot_encode).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    avg_frame_probs = np.mean([frame_model.predict(images) for images, _ in test_ds], axis=0)
    video_probs_dict = dict(zip(['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'], avg_frame_probs))

    test_set = RevDeEsDataset([audio_path], [0])
    loader, audio_probs = DataLoader(test_set, batch_size=1), []
    with torch.no_grad(): audio_probs = [torch.softmax(audio_model(x.unsqueeze(1).to(device)), dim=1).cpu().numpy()[0] for x, _ in loader]

    avg_audio_probs_raw = np.mean(audio_probs, axis=0)
    audio_probs_dict = dict(zip(['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'], avg_audio_probs_raw))

    img = Image.open(Path(audio_path).with_suffix('.png')).convert("RGB")
    input_tensor = transforms.ToTensor()(transforms.Resize((224, 224))(img)).unsqueeze(0).to(device)
    avg_spec_probs_raw = np.mean(torch.softmax(spectrogram_model(input_tensor), dim=1).detach().cpu().numpy(), axis=0)
    spec_probs_dict = dict(zip(['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'], avg_spec_probs_raw))

    video_entropy, audio_entropy, spec_entropy = map(safe_entropy, [video_probs_dict.values(), audio_probs_dict.values(), spec_probs_dict.values()])
    video_conf, audio_conf, spec_conf = 1 / (video_entropy + 1e-8), 1 / (audio_entropy + 1e-8), 1 / (spec_entropy + 1e-8)

    video_weight = (video_conf * video_acc) / (video_conf * video_acc + audio_conf * audio_acc + spec_conf * spec_acc)
    audio_weight = (audio_conf * audio_acc) / (video_conf * video_acc + audio_conf * audio_acc + spec_conf * spec_acc)
    spec_weight = (spec_conf * spec_acc) / (video_conf * video_acc + audio_conf * audio_acc + spec_conf * spec_acc)

    final_probs_dict = {emo: (video_weight * video_probs_dict.get(emo, 0.0)) + (audio_weight * audio_probs_dict.get(emo, 0.0)) + (spec_weight * spec_probs_dict.get(emo, 0.0)) for emo in video_probs_dict}

    fusion_pred = max(final_probs_dict, key=final_probs_dict.get)
    audio_pred = max(audio_probs_dict, key=audio_probs_dict.get)
    spec_pred = max(spec_probs_dict, key=spec_probs_dict.get)
    video_pred = max(video_probs_dict, key=video_probs_dict.get)

    try:
        shutil.rmtree(frame_dir)
        os.remove(audio_path)
        os.remove(Path(audio_path).with_suffix('.png'))
    except Exception as e:
        print(f"Warning: Cleanup failed - {e}")

    return fusion_pred, audio_pred, spec_pred, video_pred

dataset = pd.read_csv("mapping_train_test_valid.csv")
dataset = dataset[dataset["category"] == "valid"]
dataset[['fusion_pred', 'audio_pred', "spec_pred", 'video_pred']] = dataset['key'].apply(pred)

fusion_acc = accuracy_score(dataset["emotion_label"], dataset["fusion_pred"])
audio_acc = accuracy_score(dataset["emotion_label"], dataset["audio_pred"])
video_acc = accuracy_score(dataset["emotion_label"], dataset["video_pred"])
spec_acc = accuracy_score(dataset["emotion_label"], dataset["spec_pred"])

print(f"Fusion Accuracy: {fusion_acc}\nAudio Accuracy: {audio_acc}\nVideo Accuracy: {video_acc}\nSpec Accuracy: {spec_acc}")

for model, title in zip([dataset["fusion_pred"], dataset["audio_pred"], dataset["video_pred"]], ["Fusion", "Audio", "Video"]):
    plot_confusion_matrix(dataset['emotion_label'], model, f"{title} Model Confusion Matrix")




