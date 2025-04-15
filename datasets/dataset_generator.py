import os
import pandas as pd
from pathlib import Path

def video_paths(i):
  return f"/kaggle/input/ravdess-emotional-speech-video/RAVDESS dataset/Video_Speech_Actor_{i.split('-')[-1]}/Actor_{i.split('-')[-1]}/01-{i}.mp4"



df = pd.read_csv("mappin_audio_video.csv")

# Define the root directory for saving the dataset
dataset_root = "dataset"

# Create required directories if they don't exist
os.makedirs(os.path.join(dataset_root, 'frames'), exist_ok=True)
os.makedirs(os.path.join(dataset_root, 'audio'), exist_ok=True)
os.makedirs(os.path.join(dataset_root, 'spectrograms'), exist_ok=True)

# Iterate through DataFrame rows
for index, row in df.iterrows():
    video_path = row['paths']  # Video path
    audio_path = row['audio_paths']  # Audio path (for extraction)
    category = row['category']
    emotion_label = row['emotion_label']
    video_basename = Path(video_path).stem  # Get the base name of the video (without extension)
    
    # Directory structure for category and emotion_label
    frame_dir = os.path.join(dataset_root, 'frames', category, emotion_label)
    audio_dir = os.path.join(dataset_root, 'audio', category, emotion_label)
    spectrogram_dir = os.path.join(dataset_root, 'spectrograms', category, emotion_label)
    
    # Create directories if they don't exist
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(spectrogram_dir, exist_ok=True)
    video_path = video_paths(row["key"])
    # Extract frames, audio, and spectrograms (this assumes you have a proper function defined)
    extract_frames_and_audio_from_paths([video_path], frame_dir, os.path.dirname(audio_path), frame_rate=30)

    
    print(f"Processed video {video_basename} at {video_path}")

