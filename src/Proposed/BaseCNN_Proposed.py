#BASE CNN 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling, RandomFlip, RandomRotation
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. CONFIGURATION ---
# IMPORTANT: Update this path to where your 'Angry', 'Happy', etc. folders are located.
DATA_DIR = r"/ProposedDataset"

# Model parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 25 # You can start with 15-25 and see how the model performs

# --- 2. DATA LOADING & PREPROCESSING ---
# Keras utility to automatically load images from directories.
# It will infer class names ('Angry', etc.) from the folder names.
# We split the data into 80% for training and 20% for validation.

print("Loading training data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123, # Seed for reproducibility
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale' # Thermal images are single-channel
)

print("\nLoading validation data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

# Get the class names from the directory structure
class_names = train_ds.class_names
print(f"\nFound classes: {class_names}")
NUM_CLASSES = len(class_names)


# --- 3. DATA AUGMENTATION & NORMALIZATION LAYERS ---
# We create these as layers to make them part of the model itself.
# This ensures they are applied consistently during training and prediction.
data_augmentation = Sequential(
    [
        RandomFlip("horizontal"),
        RandomRotation(0.1),
    ]
)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# --- 4. BUILD THE CNN MODEL ---
model = Sequential([
    # Input layer: Rescale pixel values from [0, 255] to [0, 1]
    Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

    # Apply data augmentation
    data_augmentation,

    # First Convolutional Block
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(),

    # Second Convolutional Block
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(),

    # Third Convolutional Block
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(),

    # Flatten the results to feed into a dense layer
    Flatten(),

    # Dense layer for classification
    Dense(128, activation='relu'),
    Dropout(0.5), # Dropout to prevent overfitting

    # Output layer with a neuron for each class
    Dense(NUM_CLASSES, activation='softmax')
])


# --- 5. COMPILE THE MODEL ---
# We use 'SparseCategoricalCrossentropy' because our labels are integers (0, 1, 2...), not one-hot encoded.
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Print a summary of the model architecture
model.summary()


# --- 6. TRAIN THE MODEL ---
print("\nStarting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
print("\nTraining finished.")


# --- 7. VISUALIZE TRAINING RESULTS ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.suptitle('Model Training History')
plt.show()


# --- 8. SAVE THE MODEL (OPTIONAL) ---
# You can save the trained model to use it later without retraining.
model.save('thermal_emotion_model.keras')
print("\nModel saved as thermal_emotion_model.keras")


# --- 9. HOW TO MAKE A PREDICTION ON A NEW IMAGE ---
def predict_emotion(image_path):
    """Loads an image, preprocesses it, and predicts the emotion."""
    print(f"\nMaking prediction for: {image_path}")
    
    # Load the image
    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale"
    )
    
    # Convert it to a numpy array
    img_array = tf.keras.utils.img_to_array(img)
    
    # Add a batch dimension
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make the prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Print the result
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    print(
        f"This image most likely belongs to '{predicted_class}' with a {confidence:.2f}% confidence."
    )
