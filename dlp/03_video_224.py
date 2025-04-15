import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

# === SETUP ===
# Set directories for train, validation, and test data
train_data_dir = '/kaggle/working/frames/train'
val_data_dir = '/kaggle/working/frames/valid'
test_data_dir = '/kaggle/working/frames/test'

train_data_dir = pathlib.Path(train_data_dir)
val_data_dir = pathlib.Path(val_data_dir)

# Parameters
img_height, img_width = 224, 224
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

# === LOAD DATA ===
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(f"Class Names: {class_names}")

# === PREPROCESSING ===
# Normalize and one-hot encode
def one_hot_encode(batch_images, batch_labels):
    # batch_images = tf.cast(batch_images, tf.float32) / 255.0
    batch_labels_one_hot = tf.one_hot(batch_labels, depth=len(class_names))
    return batch_images, batch_labels_one_hot

# Apply to datasets with optimizations
train_ds = train_ds.map(one_hot_encode).shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(one_hot_encode).cache().prefetch(buffer_size=AUTOTUNE)

# === DATA AUGMENTATION ===
data_augmentation = tf.keras.Sequential([
    # layers.Rescaling(1./255),  # Normalize
    layers.RandomFlip("horizontal"),
    # layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

# === MODEL ===

# Pretrained ResNet50 without top layer
pretrained_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(img_height, img_width, 3),
    pooling='avg',
    weights='imagenet'
)
# Unfreeze last few layers as dataset is very different (like facial expressions, emotions, etc.), fine-tuning helps a lot.
for layer in pretrained_model.layers[-20:]:
    layer.trainable = True

# Add layers
resnet_model = Sequential([
    layers.InputLayer(input_shape=(img_height, img_width, 3)),
    data_augmentation,
    pretrained_model,
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile
optimizer = Adam(learning_rate=1e-4)
resnet_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy','f1_score'])


# === CALLBACKS ===
checkpoint_callback = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# === TRAINING ===
history = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[checkpoint_callback, early_stopping_callback]
)


## Evaluations