import os
import re
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- FIX FOR DISPLAY ERROR ---
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend (saves files instead of opening windows)
import matplotlib.pyplot as plt

# ----------------------------
# 1. CONFIGURATION
# ----------------------------
DATA_DIR = r"/home/harshan/Documents/edl/Face_expression_detection/unzipped/dataset_extracted"
IMG_SIZE = 96
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 40  # Increased slightly because learning rate is lower

# ----------------------------
# 2. LABELS & HELPERS
# ----------------------------
CODE_TO_INDEX = {
    "ha": 0, "sa": 1, "su": 2, "an": 3, 
    "fe": 4, "ne": 5, "di": 6
}

INDEX_TO_NAME = {
    0: "Happy", 1: "Sad", 2: "Surprised", 3: "Angry", 
    4: "Fear", 5: "Neutral", 6: "Disgust"
}

CODE_PATTERNS = {
    code: re.compile(rf"(?:^|[^a-z]){code}(?:[^a-z]|$)", re.IGNORECASE)
    for code in CODE_TO_INDEX.keys()
}

def get_label_from_filename(filename: str):
    name_lower = filename.lower()
    matched_codes = []
    for code, pattern in CODE_PATTERNS.items():
        if pattern.search(name_lower):
            matched_codes.append(code)

    if len(matched_codes) == 1:
        return CODE_TO_INDEX[matched_codes[0]]
    return None

# ----------------------------
# 3. DATA LOADING
# ----------------------------
print("Loading face detector...")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

X = []
y = []
skipped = 0

print(f"Scanning directory: {DATA_DIR} ...")
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

for root, dirs, files in os.walk(DATA_DIR):
    for fname in files:
        if not fname.lower().endswith(VALID_EXT):
            continue

        fpath = os.path.join(root, fname)
        label_index = get_label_from_filename(fname)
        if label_index is None:
            skipped += 1
            continue

        img = cv2.imread(fpath)
        if img is None:
            skipped += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))

        if len(faces) == 0:
            skipped += 1
            continue

        # Take largest face
        x_c, y_c, w_c, h_c = max(faces, key=lambda b: b[2] * b[3])
        face = gray[y_c : y_c + h_c, x_c : x_c + w_c]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        X.append(face)
        y.append(label_index)

# ----------------------------
# 4. PREPROCESSING & CHECKS
# ----------------------------
X = np.array(X, dtype="float32")
y = np.array(y, dtype="int32")

print(f"\nImages Loaded: {len(X)}")
print(f"Images Skipped: {skipped}")

if len(X) == 0:
    raise RuntimeError("No images loaded! Check path.")

# Check Balance
unique, counts = np.unique(y, return_counts=True)
print("\n--- CLASS DISTRIBUTION ---")
for lbl, count in zip(unique, counts):
    print(f"{INDEX_TO_NAME[lbl]}: {count} images")
print("--------------------------\n")

# Normalize
X = X / 255.0
X = np.expand_dims(X, axis=-1)

# Convert Grayscale to RGB for MobileNet (Repeat channel 3 times)
print("Converting to RGB format for MobileNet...")
X = np.repeat(X, 3, axis=-1)

# One-hot encoding
y_cat = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 5. DATA AUGMENTATION
# ----------------------------
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ----------------------------
# 6. MODEL: FINE-TUNED MOBILENET V2
# ----------------------------
print("Building Fine-Tuned MobileNetV2 model...")

# Load Base Model (Pre-trained on ImageNet)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False, 
    weights="imagenet"
)

# UNFREEZE the base model to allow Fine-Tuning
base_model.trainable = True

# Freeze the bottom 100 layers (keep basic shapes), Train top layers (learn faces)
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Add custom layers on top
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Use a VERY LOW learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# 7. TRAINING
# ----------------------------
print("\nStarting Training (This may take a while)...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS
)

# ----------------------------
# 8. SAVE & EVALUATE
# ----------------------------
save_name = "emotion_mobilenet_v2.h5"
model.save(save_name)
print(f"\nModel saved as {save_name}")

# --- SAVE TRAINING PLOT ---
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Training Accuracy")

# --- SAVE CONFUSION MATRIX ---
print("\nGenerating Confusion Matrix...")
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

# Print Report
print(classification_report(y_true, y_pred_classes, target_names=[INDEX_TO_NAME[i] for i in range(NUM_CLASSES)]))

# Plot Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[INDEX_TO_NAME[i] for i in range(NUM_CLASSES)],
            yticklabels=[INDEX_TO_NAME[i] for i in range(NUM_CLASSES)])
plt.title('Confusion Matrix')
plt.tight_layout()

# Save the final image
plt.savefig("training_results.png")
print("Results saved as 'training_results.png'. Open this file to see the graphs.")