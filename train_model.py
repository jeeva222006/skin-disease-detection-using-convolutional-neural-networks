import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os

# ── SETTINGS ──────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset/IMG_CLASSES/"
MODEL_SAVE   = "model/skin_model.h5"
LABELS_SAVE  = "model/labels.txt"
IMG_SIZE     = (64, 64)
BATCH_SIZE   = 32               # smaller batch = better learning
EPOCHS       = 20
os.makedirs("model", exist_ok=True)

# ── CHECK DATASET ──────────────────────────────────────────────────────────────
print("\n📂 Checking dataset...")
classes = sorted([f for f in os.listdir(DATASET_PATH)
                  if os.path.isdir(os.path.join(DATASET_PATH, f))])
print(f"✅ Found {len(classes)} classes:")
class_counts = {}
for c in classes:
    count = len([f for f in os.listdir(os.path.join(DATASET_PATH, c))
                 if f.lower().endswith(('.jpg','.jpeg','.png'))])
    class_counts[c] = count
    print(f"   {c}: {count} images")

# ── COMPUTE CLASS WEIGHTS (fix imbalance) ─────────────────────────────────────
total_images = sum(class_counts.values())
n_classes    = len(classes)
class_weights = {}
for idx, cls in enumerate(classes):
    # higher weight for classes with fewer images
    weight = total_images / (n_classes * class_counts[cls])
    class_weights[idx] = weight
    print(f"   Weight for {cls}: {weight:.2f}")

print(f"\n   Total images: {total_images}")

# ── DATA GENERATORS ────────────────────────────────────────────────────────────
print("\n🔄 Setting up generators...")

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = len(train_data.class_indices)
print(f"\n✅ Classes: {list(train_data.class_indices.keys())}")
print(f"✅ Steps per epoch: 100")
print(f"✅ Val steps      : 30\n")

# ── CNN MODEL ──────────────────────────────────────────────────────────────────
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation="relu", padding="same",
           input_shape=(64, 64, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    # Block 2
    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    # Block 3
    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    # Block 4
    Conv2D(256, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    # Classifier
    Flatten(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),   # lower LR = more careful learning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ── CALLBACKS ──────────────────────────────────────────────────────────────────
checkpoint = ModelCheckpoint(
    MODEL_SAVE,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ── TRAIN WITH CLASS WEIGHTS ───────────────────────────────────────────────────
print("\n🚀 Training with balanced class weights...\n")
print("Each epoch ~2-3 minutes. Total ~30-40 minutes.\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=100,
    validation_steps=30,
    class_weight=class_weights,     # ← KEY FIX: balances the dataset
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ── SAVE LABELS ────────────────────────────────────────────────────────────────
with open(LABELS_SAVE, "w") as f:
    for label in train_data.class_indices:
        f.write(label + "\n")

print(f"\n✅ Model  saved → {MODEL_SAVE}")
print(f"✅ Labels saved → {LABELS_SAVE}")
print("\n🎉 Done! Now run:  python app.py")

# ── PLOT ───────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],     label="Train", color="#0d9488", lw=2)
plt.plot(history.history["val_accuracy"], label="Val",   color="#f59e0b", lw=2)
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(history.history["loss"],     label="Train", color="#0d9488", lw=2)
plt.plot(history.history["val_loss"], label="Val",   color="#f59e0b", lw=2)
plt.title("Loss per Epoch")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("model/training_results.png", dpi=120)
plt.show()
print("✅ Plot saved → model/training_results.png")