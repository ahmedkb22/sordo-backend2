import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import json

# =========================
# CONFIG
# =========================
PROCESSED_PATH  = "../wlasl/processed"
MODEL_OUTPUT    = "../models/sign_model.h5"
LABELS_OUTPUT   = "../models/labels.json"
SEQUENCE_LENGTH = 30
AUGMENT_FACTOR  = 4     # each real sample produces this many augmented copies

# =========================
# LOAD DATA
# =========================
print("📦 Loading data...")

X, y   = [], []
labels = sorted(os.listdir(PROCESSED_PATH))

print(f"📋 Found words: {labels}")

for label_idx, label in enumerate(labels):
    label_path = os.path.join(PROCESSED_PATH, label)
    files = [f for f in os.listdir(label_path) if f.endswith(".npy")]
    print(f"  → {label}: {len(files)} samples")

    for fname in files:
        seq = np.load(os.path.join(label_path, fname))
        if seq.shape == (SEQUENCE_LENGTH, 126):
            X.append(seq)
            y.append(label_idx)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\n✅ Total samples loaded: {len(X)}")
print(f"✅ Shape: {X.shape}")

# =========================
# AUGMENTATION
# Richer than simple noise — also applies temporal jitter and scaling.
# =========================
def augment(seq):
    """Return one randomly augmented copy of seq (30, 126)."""
    out = seq.copy()

    choice = np.random.randint(0, 3)

    if choice == 0:
        # Gaussian noise
        out += np.random.normal(0, 0.015, out.shape).astype(np.float32)

    elif choice == 1:
        # Random per-hand scale (simulate different hand sizes / distances)
        for slot in range(2):
            s = np.random.uniform(0.88, 1.12)
            out[:, slot * 63:(slot + 1) * 63] *= s

    else:
        # Temporal shift: roll frames by 1-3 and pad with zeros
        shift = np.random.randint(1, 4)
        out   = np.roll(out, shift, axis=0)
        out[:shift] = 0.0

    return out

print("\n🔧 Augmenting data...")

X_aug, y_aug = list(X), list(y)

for i in range(len(X)):
    for _ in range(AUGMENT_FACTOR):
        X_aug.append(augment(X[i]))
        y_aug.append(y[i])

X_aug = np.array(X_aug, dtype=np.float32)
y_aug = np.array(y_aug)

print(f"✅ After augmentation: {len(X_aug)} samples")

# =========================
# PREPARE
# =========================
num_classes = len(labels)
y_cat       = to_categorical(y_aug, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_cat, test_size=0.2, random_state=42, stratify=y_aug
)

print(f"\n🏋️  Train: {len(X_train)} | Test: {len(X_test)}")

# Class weights — compensate for imbalanced word counts in WLASL
raw_classes = np.argmax(y_train, axis=1)
cw_values   = compute_class_weight("balanced", classes=np.unique(raw_classes), y=raw_classes)
class_weight = dict(enumerate(cw_values))
print(f"⚖️  Class weights: { {labels[k]: round(v, 2) for k, v in class_weight.items()} }")

# =========================
# BUILD MODEL
# Large LSTM first (captures long-range temporal deps),
# then smaller LSTM, then MLP head with BatchNorm.
# =========================
print("\n🧠 Building LSTM model...")

model = Sequential([
    # First LSTM: large, sees full sequence
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 126)),
    Dropout(0.3),

    # Second LSTM: smaller, distils into a fixed vector
    LSTM(64, return_sequences=False),
    Dropout(0.3),

    # MLP head
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu'),
    BatchNormalization(),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# CALLBACKS
# =========================
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=25,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# =========================
# TRAIN
# =========================
print("\n🚀 Training...")

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,          # larger batch = stabler gradients
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight
)

# =========================
# EVALUATE
# =========================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy : {accuracy * 100:.2f}%")
print(f"   Test Loss     : {loss:.4f}")

# Per-class accuracy
y_pred  = model.predict(X_test, verbose=0)
y_true  = np.argmax(y_test,  axis=1)
y_hat   = np.argmax(y_pred,  axis=1)

print("\n📊 Per-word accuracy:")
for i, label in enumerate(labels):
    mask  = y_true == i
    if mask.sum() == 0:
        continue
    acc_i = np.mean(y_hat[mask] == i)
    bar   = "█" * int(acc_i * 20)
    print(f"   {label:10s}  {bar:<20s}  {acc_i * 100:.1f}%  ({mask.sum()} samples)")

# =========================
# SAVE
# =========================
os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)

model.save(MODEL_OUTPUT)
print(f"\n💾 Model saved : {MODEL_OUTPUT}")

with open(LABELS_OUTPUT, "w") as f:
    json.dump(labels, f)
print(f"💾 Labels saved: {LABELS_OUTPUT}")

print("\n🎉 Training complete!")