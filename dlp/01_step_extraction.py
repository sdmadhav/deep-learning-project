import os
from pathlib import Path
import cv2
import moviepy.editor as mp

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