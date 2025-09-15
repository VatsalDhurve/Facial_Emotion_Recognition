#
# Emotion Classification from Thermal Images using Transfer Learning
#
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os


DATA_DIR = r"Dataset" 


if not os.path.exists(DATA_DIR):
    print(f"Error: The directory '{DATA_DIR}' does not exist.")
    print("Please update the 'DATA_DIR' variable to the correct path of your dataset.")
    exit()

IMAGE_SIZE = (224, 224) # Image dimensions for VGG16
BATCH_SIZE = 32         # Number of images to process in a batch
EPOCHS = 25             # Maximum number of training cycles
CLASSES = sorted(os.listdir(DATA_DIR)) # Automatically get class names from folder names
NUM_CLASSES = len(CLASSES)
print(f"Found {NUM_CLASSES} classes: {CLASSES}")


# --- 2. DATA LOADING & PREPROCESSING ---
print("\n--- Loading and Preprocessing Data ---")

# Load dataset from directories, splitting into 80% training and 20% validation
# We set color_mode='rgb' to duplicate the single grayscale channel into three channels,
# which is what the pre-trained VGG16 model expects as input.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    color_mode='rgb' 
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    color_mode='rgb'
)

# Create a data augmentation layer to prevent overfitting
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Apply augmentation and VGG16-specific preprocessing to the datasets
# The VGG16 preprocessing function handles pixel scaling and format adjustments.
def prepare_dataset(ds, augment=False):
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y) if augment else (x, y), 
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.keras.applications.vgg16.preprocess_input(x), y), 
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = prepare_dataset(train_ds, augment=True)
val_ds = prepare_dataset(val_ds, augment=False)

print("✅ Data loaded and prepared successfully!")


# --- 3. MODEL BUILDING (Transfer Learning) ---
print("\n--- Building Model using Transfer Learning (VGG16) ---")

# Load the pre-trained VGG16 model without its classification layers
base_model = VGG16(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,  # Exclude the final dense layers of ImageNet
    weights='imagenet'
)

# Freeze the convolutional layers of the base model
base_model.trainable = False

# Create a new model by adding our custom classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(), # Efficiently flattens the feature maps
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Add dropout for regularization to reduce overfitting
    layers.Dense(NUM_CLASSES, activation='softmax') # Output layer for our emotion classes
])

# Compile the model with an optimizer, loss function, and metrics
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# --- 4. MODEL TRAINING ---
print("\n--- Starting Model Training ---")

# Define callbacks to improve training
# EarlyStopping stops training if performance on the validation set worsens
# ModelCheckpoint saves the best version of the model during training
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_emotion_classifier.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("🎉 Model training complete!")


# --- 5. PERFORMANCE EVALUATION ---
print("\n--- Evaluating Model Performance ---")

# Plot training & validation accuracy and loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
plt.show()

# Generate predictions on the validation dataset to create a confusion matrix
print("\nGenerating predictions on validation data...")
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_true = np.argmax(y_true, axis=1)

# Create and display the confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.show()

# Print a detailed classification report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASSES))

print("\n--- Script Finished ---")
