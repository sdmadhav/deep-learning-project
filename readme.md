# Emotion Classification from Multimodal Data

This repository contains a deep learning project aimed at emotion classification using multiple data modalities, including text, audio, and video. The goal is to classify emotions from a variety of data sources using different machine learning models and techniques.

## Project Structure

The project consists of the following main directories and files:

### `datasets/`
- **`text_emotion_dataset.csv`**: The main dataset used for training and testing models. It contains labeled text data for emotion classification.
- **`dataset_generator.py`**: Script to generate the dataset, preprocess the data, and split it into training, validation, and test sets.
- **`mapping_train_test_valid.csv`**: CSV file containing the mapping of data samples to train, test, and validation splits.

### `saved_models/`
- **`best_spectrogram_cnn.pth`**: The best model for emotion classification using spectrograms (CNN-based).
- **`crnn_attention_final_model.pth`**: Final trained model for emotion classification using CRNN with attention mechanism.
- **`best_model(1).keras`**: Keras model for emotion classification, trained on various features from the dataset.

### `results/`
- **`Naive Bayes Model_confusion_matrix.png`**: Confusion matrix for the Naive Bayes model.
- **`Random Forest Model_confusion_matrix.png`**: Confusion matrix for the Random Forest model.
- **`evaluation_metrics.png`**: Visualization of evaluation metrics for various models.
- **`classification_report.txt`**: Text file containing the classification report for different models.
- **`confusion_matrix(1).png`**: Confusion matrix for a model’s predictions.
- **`emotion_dataset(1).csv`**: CSV file containing a cleaned version of the emotion dataset.
- **`Decision Tree Model_confusion_matrix.png`**: Confusion matrix for the Decision Tree model.
- **`test_profile(2).html`**: Profiling of the test dataset using Sweetviz.
- **`sweetviz_train_vs_test(2).html`**: Sweetviz report comparing train vs test datasets.
- **`train_profile(2).html`**: Profiling of the training dataset using Sweetviz.
- **`sample_submission(1).csv`**: Sample submission format for the project.
- **`sweetviz_train_vs_val(2).html`**: Sweetviz report comparing train vs validation datasets.
- **`confusion_matrix.png`**: Overall confusion matrix.
- **`evaluation_metrics(1).png`**: Visualized evaluation metrics for the model.
- **`01-01-05-01-01-01-06_audio.png`**: Audio feature visualization for one of the samples.
- **`val_profile(2).html`**: Profiling of the validation dataset using Sweetviz.

### `dlp/`
- **`05_final_fusion_model.py`**: Script for the final fusion model combining different modalities for emotion classification.
- **`emotion_text_classification.py`**: Script to perform text-based emotion classification using different algorithms.
- **`allin_one.py`**: Unified script combining all preprocessing, feature extraction, and model training into one.
- **`04_spectrogram.py`**: Script for spectrogram generation from audio data.
- **`02_audio.py`**: Script for audio preprocessing and feature extraction.
- **`00_train_test_split.py`**: Script for splitting the dataset into training and testing sets.
- **`01_step_extraction.py`**: Feature extraction script for the dataset.
- **`03_video_224.py`**: Video processing script for extracting frames for emotion classification.

### `notebooks/`
- **`data-engineering-and-model-devlopment(3).ipynb`**: Jupyter notebook for data engineering and initial model development.
- **`Final_Fusion_Model.ipynb`**: Jupyter notebook for building and evaluating the final fusion model.
- **`data engineering and model development.ipynb`**: Another notebook detailing the engineering steps and model development process.

### `DL_project slides_(142402008&142402010).pdf`
- **`slides_(142402008&142402010).pdf`**: Project presentation slides containing an overview of the methods and results.

## Installation

To run the project, ensure you have the following dependencies installed:

- Python 3.7+
- PyTorch (for deep learning models)
- Keras (for Keras models)
- TensorFlow (if using Keras models)
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Sweetviz (for data visualization)

You can install the required libraries via `pip`:

```bash
pip install -r requirements.txt

