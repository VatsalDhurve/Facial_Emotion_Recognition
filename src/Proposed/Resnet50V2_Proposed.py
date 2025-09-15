#manual weights 2.0
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Lambda
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# --- 1. CONFIGURATION ---
# IMPORTANT: Update this path to where your emotion folders are located.
DATA_DIR = r"\Proposed Dataset"

# Model parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# Set a high number of epochs; EarlyStopping will find the best number automatically.
INITIAL_EPOCHS = 50
FINE_TUNE_EPOCHS = 20

# --- 2. DATA LOADING & PREPROCESSING ---
print("Loading training data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='rgb'
)

print("\nLoading validation data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='rgb'
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"\nFound classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. BUILD THE RESNET50V2 MODEL ---
inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = Lambda(preprocess_input)(inputs)
base_model = ResNet50V2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs, outputs)

# --- 4. COMPILE THE MODEL AND DEFINE CALLBACKS ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.summary()

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)

# --- 5. MANUALLY DEFINE CLASS WEIGHTS ---
print("\nDefining manual class weights to find a better balance...")

# Tune these values to find the best balance for your needs.
# A higher value tells the model to pay more attention to that class.
manual_weights = {
    'Sad': 2.5,      # Give a strong boost
    'Fear': 2.5,     # Give a strong boost
    'Surprise': 2.0, # Give a moderate boost
    'Angry': 1.5,    # Give a slight boost
    'Disgust': 1.0,  # No boost (majority class)
    'Happy': 1.0     # No boost (majority class)
}

# Create the dictionary that Keras expects (integer index -> weight)
class_weight_dict = {i: manual_weights.get(class_name, 1.0) for i, class_name in enumerate(class_names)}
print(f"Using Manual Class Weights: {class_weight_dict}")


# --- 6. INITIAL TRAINING ---
print("\n--- Starting Initial Training (Classifier Head Only) ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[early_stopping_callback]
)

# --- 7. FINE-TUNING THE MODEL ---
print("\n--- Starting Fine-Tuning (Unfreezing Top Layers) ---")
base_model.trainable = True
for layer in base_model.layers[:150]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.summary()

initial_fine_tune_epoch = len(history.epoch)
total_epochs = initial_fine_tune_epoch + FINE_TUNE_EPOCHS

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=initial_fine_tune_epoch,
    validation_data=val_ds,
    class_weight=class_weight_dict,
    callbacks=[early_stopping_callback]
)

# --- 8. VISUALIZE COMBINED TRAINING HISTORY ---
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(x=initial_fine_tune_epoch -1, color='gray', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='lower right')
plt.title('Combined Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(x=initial_fine_tune_epoch -1, color='gray', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='upper right')
plt.title('Combined Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# --- 9. GENERATE FINAL CLASSIFICATION REPORT ---
print("\nGenerating final classification report for validation data...")
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# --- 10. GENERATE AND DISPLAY CONFUSION MATRIX ---
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# --- 11. SAVE THE FINAL MODEL ---
model.save('thermal_emotion_resnet_manual_weights_model.keras')
print("\nFinal model saved as thermal_emotion_resnet_manual_weights_model(1).keras")
