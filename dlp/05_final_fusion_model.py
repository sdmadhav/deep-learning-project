# -*- coding: utf-8 -*-
"""Final_Fusion_Model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15GiSHCVWulGGh4KbUgKdIcbsiCjq6Npf
"""

# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
# kagglehub.login()

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

adrivg_ravdess_emotional_speech_video_path = kagglehub.dataset_download('adrivg/ravdess-emotional-speech-video')
# madhavsdeshatwad_delete_this_files_path = kagglehub.dataset_download('madhavsdeshatwad/delete-this-files')
# madhavsdeshatwad_audio_classifier_for_revdees_pytorch_default_1_path = kagglehub.model_download('madhavsdeshatwad/audio-classifier-for-revdees/PyTorch/default/1')
# madhavsdeshatwad_video_frame_emotion_classifier_revdees_keras_default_1_path = kagglehub.model_download('madhavsdeshatwad/video-frame-emotion-classifier-revdees/Keras/default/1')

print('Data source import complete.')

adrivg_ravdess_emotional_speech_video_path

"""Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
# https://www.kaggle.com/datasets/adrivg/ravdess-emotional-speech-video
"""

import cv2
import moviepy.editor as mp
from pathlib import Path
import torch
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from torchvision import transforms
import os
from glob import glob
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import pathlib
from sklearn.metrics import accuracy_score, f1_score
import random
import librosa
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def extract_frames_and_audio_from_paths(video_paths, output_folder, audio_folder, frame_rate=1):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    audio_folder = Path(audio_folder)
    audio_folder.mkdir(parents=True, exist_ok=True)

    for index, video_path in enumerate(video_paths):
        print(f"Processing video {index + 1}: {video_path}")

        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            continue

        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            continue

        video_name = video_path.stem  # filename without extension
        count = 0
        saved = 0

        # Extract frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_rate == 0:
                frame_filename = output_folder / f"{video_name}_frame_{saved:04d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                saved += 1

            count += 1

        cap.release()
        print(f"Extracted {saved} frames from {video_path.name}")

        # Extract audio using moviepy
        try:
            my_clip = mp.VideoFileClip(str(video_path))
            # Ensure the audio path is relative to the current directory or provide the full path
            audio_path = audio_folder / f"{video_name}_audio.mp3"
            my_clip.audio.write_audiofile(str(audio_path))
            print(f"Audio saved to {audio_path}")

            def save_spectrogram(audio_path, save_path, sr=22050, n_mels=128):
                import librosa
                import librosa.display
                import matplotlib.pyplot as plt
                import numpy as np
                y, _ = librosa.load(audio_path, sr=sr)
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                S_DB = librosa.power_to_db(S, ref=np.max)

                plt.figure(figsize=(3, 3))
                librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
                plt.axis('off')
                plt.tight_layout()
                # Ensure save_path is either relative or a full path
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()

            # Ensure the save path is either relative or a full path
            save_spectrogram(audio_path, audio_path.with_suffix('.png'))

        except Exception as e:
            print(f"Failed to extract audio from {video_path.name}: {e}")

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
    def __init__(self, file_paths, Augment, labels=None, sr=16000, duration=3):
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

# Check for GPU (CUDA) availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

audio_model = CRNN(input_shape=(94, 129), num_classes=8).to(device)

# Load the checkpoint
checkpoint = torch.load('/content/crnn_attention_final_model.pth', weights_only=False,
                        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Extract the model state_dict from the checkpoint
model_state_dict = checkpoint['model_state_dict']

# Now load the model state_dict into the model
audio_model.load_state_dict(model_state_dict)

criterion = nn.CrossEntropyLoss()





import torchvision.models as models

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SpectrogramCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load models
# Load models
data = ""
# Recursive function to remove "value_range"
def remove_value_range(obj):
    !unzip /content/best_model.keras -d model_dir
    import json

    # Path to your config.json
    config_path = "/content/model_dir/config.json"

    # Load the JSON file
    with open(config_path, "r") as f:
        data = json.load(f)
    if isinstance(obj, dict):
        # Remove 'value_range' key if exists
        if "value_range" in obj:
            del obj["value_range"]
            print("✅ 'value_range' removed successfully.")
        # Recurse into dictionary values
        for key in obj:
            remove_value_range(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            remove_value_range(item)

# Apply the function to the loaded JSON
# remove_value_range(data)

# Save the modified JSON back to file
# with open(config_path, "w") as f:
#     json.dump(data, f, indent=4)

print("✅ Removed all 'value_range' entries from config.json.")

## replace !unzip /content/best_model.keras -d model_dir
# Open model_dir/keras_metadata.pb or model.json (whichever exists) and manually remove the "value_range": [0, 255] entry.
frame_model =  tf.keras.models.load_model("/content/best_model(1).keras")
audio_model   #  CRNN + Attention model is already loaded
audio_model.eval()

spectrogram_model = SpectrogramCNN(num_classes=8).to(device)
spectrogram_model.load_state_dict(torch.load("best_spectrogram_cnn.pth",  map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
spectrogram_model.eval()

frame_model.summary()

import shutil
from torchvision import datasets

# Class names (Video index order)
video_class_names = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Class names (Audio index order)
audio_class_names = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


def predict_from_video(video_path, frame_dir, audio_path, frame_model, audio_model, device,
                       video_acc=0.9146, audio_acc=0.9236):
    # --- Step 1: Extract frames and audio ---
    extract_frames_and_audio_from_paths([video_path], frame_dir+"/test/", os.path.dirname(audio_path), frame_rate=30)

    # --- Step 2: Predict from frames ---
    # === PREPROCESSING ===
    # === SETUP ===

    # Parameters
    test_data_dir = pathlib.Path("/content/extracted_frames")
    # test_data_dir = pathlib.Path("/content/test")
    img_height, img_width = 224, 224
    batch_size = 32
    AUTOTUNE = tf.data.AUTOTUNE

    # Load data
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Normalize and one-hot encode
    def one_hot_encode(batch_images, batch_labels):
        # batch_images = tf.cast(batch_images, tf.float32) / 255.0
        batch_labels_one_hot = tf.one_hot(batch_labels, depth=8)
        return batch_images, batch_labels_one_hot

    test_ds = test_ds.map(one_hot_encode).cache().prefetch(buffer_size=AUTOTUNE)

    # Predict on validation data
    y_true, y_pred = [], []
    frame_probs = []

    for images, labels in test_ds:
        preds = frame_model.predict(images)
        frame_probs.append(preds)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    # Aggregate frame predictions
    frame_probs_concat = np.concatenate(frame_probs, axis=0)
    avg_frame_probs = np.mean(frame_probs_concat, axis=0)

    # Map predictions to class names
    video_probs_dict_raw = dict(zip(video_class_names, avg_frame_probs))
    video_probs_dict = {emo: video_probs_dict_raw.get(emo, 0.0) for emo in video_class_names}


    # # --- Step 3: Predict from audio ---
    test_set = RevDeEsDataset([audio_path], [0])
    loader = DataLoader(test_set, batch_size=1)
    audio_probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device).unsqueeze(1)
            out = audio_model(x)
            prob = torch.softmax(out, dim=1).cpu().numpy()[0]
            audio_probs.append(prob)

    avg_audio_probs_raw = np.mean(audio_probs, axis=0)
    audio_probs_dict_raw = dict(zip(audio_class_names, avg_audio_probs_raw))
    audio_probs_dict = {emo: audio_probs_dict_raw.get(emo, 0.0) for emo in audio_class_names}



    # ### Predict from Spectrogram

    # Image preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    from PIL import Image
    audio_path = Path(audio_path)
    image_path = audio_path.with_suffix('.png')
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    out = spectrogram_model(input_tensor)
    out = torch.softmax(out, dim=1)  # Now each row sums to 1

    # 🔄 Detach the tensor before converting to NumPy array
    avg_spec_probs_raw = np.mean(out.detach().cpu().numpy(), axis=0)  # Convert the PyTorch tensor to a NumPy array before calculating the mean.spec_probs_dict_raw = dict(zip(audio_class_names, avg_spec_probs_raw))
    spec_probs_dict_raw = dict(zip(audio_class_names, avg_spec_probs_raw))
    spec_probs_dict = {emo: spec_probs_dict_raw.get(emo, 0.0) for emo in audio_class_names}

    # --- Step 4: Weighted Fusion ---
    from scipy.stats import entropy

    # Convert dicts to arrays
    video_probs_arr = np.array([float(np.mean(v)) if isinstance(v, np.ndarray) else float(v) for v in video_probs_dict.values()])
    audio_probs_arr = np.array([float(np.mean(v)) if isinstance(v, np.ndarray) else float(v) for v in audio_probs_dict.values()])
    spec_probs_arr = np.array([float(np.mean(v)) if isinstance(v, np.ndarray) else float(v) for v in spec_probs_dict.values()])



    def safe_entropy(probs, eps=1e-6):
        probs = np.array(probs)
        probs = probs / (np.sum(probs) + eps)  # Normalize
        return entropy(probs + eps)  # Add epsilon to avoid log(0)


    # Compute entropy (uncertainty)
    video_entropy = safe_entropy(video_probs_arr)
    audio_entropy = safe_entropy(audio_probs_arr)
    spec_entropy = safe_entropy(spec_probs_arr)

    # Model performed on validation
    video_acc = 0.69
    audio_acc = 0.70
    spec_acc = 0.79

    # Calculate confidence (per-sample)
    video_conf = 1 / (video_entropy + 1e-8)
    audio_conf = 1 / (audio_entropy + 1e-8)
    spec_conf = 1 / (spec_entropy + 1e-8)

    # Combine: trust * confidence
    video_score = video_conf * video_acc
    audio_score = audio_conf * audio_acc
    spec_score = spec_conf * spec_acc

    # Normalize weights
    total_score = video_score + audio_score + spec_score
    video_weight = video_score / total_score
    audio_weight = audio_score / total_score
    spec_weight = spec_score / total_score

    print("Per-sample weights -> video:", video_weight, "audio:", audio_weight, 'spectrogram', spec_weight )

    final_probs_dict = {
        emo: (video_weight * video_probs_dict.get(emo, 0.0) +
              audio_weight * audio_probs_dict.get(emo, 0.0)) + (spec_weight * spec_probs_dict.get(emo, 0.0))
        for emo in video_class_names
    }


    # --- Step 5: Predict Top Classes ---
    # Convert all values to float by averaging if it's an array
    final_probs_list = [float(np.mean(v)) if isinstance(v, np.ndarray) else float(v) for v in final_probs_dict.values()]
    fusion_pred = video_class_names[np.argmax(final_probs_list)]

    audio_probs_list = [float(np.mean(v)) if isinstance(v, np.ndarray) else float(v) for v in audio_probs_dict.values()]
    audio_pred = audio_class_names[np.argmax(audio_probs_list)]

    spec_probs_list = [float(np.mean(v)) if isinstance(v, np.ndarray) else float(v) for v in spec_probs_dict.values()]
    spec_pred = audio_class_names[np.argmax(spec_probs_list)]


    video_probs_list = [float(np.mean(v)) if isinstance(v, np.ndarray) else float(v) for v in video_probs_dict.values()]
    video_pred = video_class_names[np.argmax(video_probs_list)]


    # --- Step 6: Cleanup ---
    try:
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(image_path):
            os.remove(image_path)
    except Exception as e:
        print(f"Warning: Cleanup failed - {e}")

    return  fusion_pred, audio_pred, spec_pred, video_pred #
    return "pass","pass","pass","pass"

def pred(i, revdess=True):
    frame_dir = "/content/extracted_frames"
    if revdess:
        video_file = f"/kaggle/input/ravdess-emotional-speech-video/RAVDESS dataset/Video_Speech_Actor_{i.split('-')[-1]}/Actor_{i.split('-')[-1]}/01-{i}.mp4"
        audio_path = f"extracted_audio/01-{i}_audio.mp3"
    else:
        video_file = i
        file_name = os. path. basename(i)
        file_name = file_name.split(".")[0]
        audio_path =  f"extracted_audio/{file_name}_audio.mp3"
    fusion_pred, audio_pred, spec_pred, video_pred = predict_from_video(video_file, frame_dir, audio_path, frame_model, audio_model, device)

    return pd.Series({
        "fusion_pred": fusion_pred,
        "audio_pred": audio_pred,
        "spectrogram_pred": spec_pred,
        "video_pred": video_pred
    })

pred("/kaggle/input/ravdess-emotional-speech-video/RAVDESS dataset/Video_Speech_Actor_01/Actor_01/01-01-01-01-01-01-01.mp4", False)

import pandas as pd
dataset = pd.read_csv("mapping_train_test_valid.csv")

dataset = dataset[dataset["category"]=="valid"]

dataset.columns

dataset[['fusion_pred', 'audio_pred', "spec_pred", 'video_pred']] = dataset['key'].apply(pred)

# Now compare with true emotion
from sklearn.metrics import accuracy_score

fusion_acc = accuracy_score(dataset["emotion_label"], dataset["fusion_pred"])

print(f"Fusion Accuracy: {fusion_acc}")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os


# Save confusion matrices as image files
def save_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title(filename, fontsize=16)
    plt.grid(False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename + ".png") # Save the figure
    plt.close(fig) # Close the figure to free up memory

# Create the "confusion_matrices" directory if it doesn't exist
os.makedirs("confusion_matrices", exist_ok=True)

save_confusion_matrix(dataset['emotion_label'], dataset['fusion_pred'], os.path.join("confusion_matrices", "Fusion_Model_Confusion_Matrix"))
save_confusion_matrix(dataset['emotion_label'], dataset['audio_pred'], os.path.join("confusion_matrices", "Audio_Model_Confusion_Matrix"))
save_confusion_matrix(dataset['emotion_label'], dataset['video_pred'], os.path.join("confusion_matrices", "Video_Model_Confusion_Matrix"))
save_confusion_matrix(dataset['emotion_label'], dataset['spec_pred'], os.path.join("confusion_matrices", "Spectrogram_Model_Confusion_Matrix"))

print("Confusion matrices saved to confusion_matrices directory.")

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Now calculate and print the classification reports
console.print("Fusion Classification Report:\n", classification_report(dataset["emotion_label"], dataset["fusion_pred"], target_names=class_names))
console.print("Audio Classification Report:\n", classification_report(dataset["emotion_label"], dataset["audio_pred"], target_names=class_names))
console.print("Video Classification Report:\n", classification_report(dataset["emotion_label"], dataset["video_pred"], target_names=class_names))
console.print("Spectrogram Classification Report:\n", classification_report(dataset["emotion_label"], dataset["spec_pred"], target_names=class_names))



