import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import json

# =========================
# CONFIG
# =========================
PROCESSED_PATH = "../wlasl/processed"
MODEL_OUTPUT = "../models/sign_model.h5"
LABELS_OUTPUT = "../models/labels.json"
SEQUENCE_LENGTH = 30

# =========================
# LOAD DATA
# =========================
print("📦 Loading data...")

X = []
y = []
labels = sorted(os.listdir(PROCESSED_PATH))

print(f"📋 Found words: {labels}")

for label_idx, label in enumerate(labels):
    label_path = os.path.join(PROCESSED_PATH, label)
    files = [f for f in os.listdir(label_path) if f.endswith(".npy")]
    
    print(f"  → {label}: {len(files)} samples")
    
    for file in files:
        file_path = os.path.join(label_path, file)
        sequence = np.load(file_path)
        
        # Ensure correct shape
        if sequence.shape == (SEQUENCE_LENGTH, 126):
            X.append(sequence)
            y.append(label_idx)

X = np.array(X)
y = np.array(y)

print(f"\n✅ Total samples: {len(X)}")
print(f"✅ Shape: {X.shape}")

# =========================
# DATA AUGMENTATION
# =========================
print("\n🔧 Augmenting data...")

def augment(sequence):
    # Add small random noise
    noise = np.random.normal(0, 0.01, sequence.shape)
    return sequence + noise

X_aug = []
y_aug = []

for i in range(len(X)):
    X_aug.append(X[i])
    y_aug.append(y[i])
    # Add 3 augmented copies
    for _ in range(3):
        X_aug.append(augment(X[i]))
        y_aug.append(y[i])

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)

print(f"✅ After augmentation: {len(X_aug)} samples")

# =========================
# PREPARE
# =========================
num_classes = len(labels)
y_cat = to_categorical(y_aug, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_cat, test_size=0.2, random_state=42
)

print(f"\n🏋️ Train: {len(X_train)} | Test: {len(X_test)}")

# =========================
# BUILD LSTM MODEL
# =========================
print("\n🧠 Building LSTM model...")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 126)),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# TRAIN
# =========================
print("\n🚀 Training...")

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# =========================
# EVALUATE
# =========================
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

# =========================
# SAVE
# =========================
os.makedirs("../models", exist_ok=True)

model.save(MODEL_OUTPUT)
print(f"💾 Model saved: {MODEL_OUTPUT}")

with open(LABELS_OUTPUT, "w") as f:
    json.dump(labels, f)
print(f"💾 Labels saved: {LABELS_OUTPUT}")

print("\n🎉 Training complete!")