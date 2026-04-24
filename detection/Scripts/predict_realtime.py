import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import json
import collections

# =========================
# CONFIG
# =========================
MODEL_PATH           = "../models/sign_model.h5"
LABELS_PATH          = "../models/labels.json"
SEQUENCE_LENGTH      = 30
PREDICTION_THRESHOLD = 0.60
BUFFER_SIZE          = 15   # smoothing window (was 5 — too jumpy)
CONFIRM_VOTES        = 8    # how many of the last BUFFER_SIZE must agree

# =========================
# LOAD MODEL & LABELS
# =========================
model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

print(f"✅ Model loaded | Words: {labels}")

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,   # matches collect_data.py
    min_tracking_confidence=0.5
)

# =========================
# NORMALISATION  ← must mirror collect_data.py exactly
# =========================
def normalize_landmarks(raw_126):
    arr = np.array(raw_126, dtype=np.float32).reshape(2, 21, 3)
    for h in range(2):
        if np.any(arr[h] != 0):
            arr[h] -= arr[h, 0, :]                 # centre on wrist
            scale = np.max(np.abs(arr[h])) + 1e-6
            arr[h] /= scale
    return arr.flatten().tolist()

# =========================
# CONSISTENT HAND ASSIGNMENT  ← must mirror collect_data.py exactly
# slot 0 = right hand, slot 1 = left hand
# =========================
def assign_hands(result):
    landmarks  = [0.0] * 126
    handedness = result.multi_handedness if result.multi_handedness else []

    for idx, hand_lms in enumerate(result.multi_hand_landmarks[:2]):
        label = handedness[idx].classification[0].label if idx < len(handedness) else "Right"
        slot  = 0 if label == "Right" else 1

        for j, lm in enumerate(hand_lms.landmark):
            base = slot * 63 + j * 3
            landmarks[base]     = lm.x
            landmarks[base + 1] = lm.y
            landmarks[base + 2] = lm.z

    return normalize_landmarks(landmarks)

# =========================
# STATE
# =========================
sequence            = []
predictions_buffer  = collections.deque(maxlen=BUFFER_SIZE)
prob_buffer         = collections.deque(maxlen=BUFFER_SIZE)  # averaged softmax
current_word        = ""
current_confidence  = 0.0
frame_counter       = 0
no_hand_frames      = 0
NO_HAND_RESET_AFTER = 10   # frames of missing hand before resetting state

# =========================
# WEBCAM LOOP
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("🎥 Webcam started — show a sign!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    frame     = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result    = hands.process(frame_rgb)

    # ── Landmark extraction ───────────────────────────────────────────────
    if result.multi_hand_landmarks:
        no_hand_frames = 0

        for hand_lms in result.multi_hand_landmarks[:2]:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        frame_lms = assign_hands(result)
        sequence.append(frame_lms)
        sequence = sequence[-SEQUENCE_LENGTH:]

    else:
        no_hand_frames += 1

        # Grace period: keep the sequence alive briefly so a stray missed
        # detection doesn't wipe good data. Reset only after N consecutive
        # frames with no hand.
        if no_hand_frames >= NO_HAND_RESET_AFTER:
            sequence            = []
            current_word        = ""
            current_confidence  = 0.0
            predictions_buffer.clear()
            prob_buffer.clear()
            frame_counter       = 0
            no_hand_frames      = 0

    # ── Predict every 3 frames (was 5 — slightly more responsive) ─────────
    if len(sequence) == SEQUENCE_LENGTH and frame_counter % 3 == 0:
        input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
        probs      = model.predict(input_data, verbose=0)[0]

        prob_buffer.append(probs)

        # Average probabilities over the buffer for smoother decisions
        avg_probs      = np.mean(prob_buffer, axis=0)
        predicted_idx  = np.argmax(avg_probs)
        confidence     = avg_probs[predicted_idx]
        predicted_word = labels[predicted_idx]

        if confidence >= PREDICTION_THRESHOLD:
            predictions_buffer.append(predicted_word)

            # Only surface a word once enough votes agree
            votes = predictions_buffer.count(predicted_word)
            if votes >= CONFIRM_VOTES:
                current_word       = predicted_word
                current_confidence = float(confidence)
        else:
            predictions_buffer.clear()
            prob_buffer.clear()
            current_word       = ""
            current_confidence = float(confidence)

    # ── Display ───────────────────────────────────────────────────────────
    h, w, _ = frame.shape

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)

    if current_word:
        text  = f"{current_word.upper()}  {int(current_confidence * 100)}%"
        color = (0, 255, 100)
        size  = 1.4
        cv2.putText(frame, text, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, 3)

    elif result.multi_hand_landmarks:
        if current_confidence > 0:
            # Show the top-2 candidates while analysing
            if len(prob_buffer):
                avg = np.mean(prob_buffer, axis=0)
                top2_idx = np.argsort(avg)[::-1][:2]
                hint = "  /  ".join(
                    f"{labels[i]} {int(avg[i]*100)}%"
                    for i in top2_idx if avg[i] > 0.15
                )
                cv2.putText(frame, hint or "Analysing…", (20, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 255), 2)
            else:
                cv2.putText(frame, "Analysing…", (20, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2)
        else:
            cv2.putText(frame, "Analysing…", (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2)

    else:
        cv2.putText(frame, "Show a sign…", (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 80, 80), 2)

    # Sequence fill bar (bottom of screen) — shows how full the buffer is
    fill_ratio = len(sequence) / SEQUENCE_LENGTH
    bar_w      = int(w * fill_ratio)
    bar_color  = (0, 200, 255) if fill_ratio < 1.0 else (0, 255, 100)
    cv2.rectangle(frame, (0, h - 8), (bar_w, h), bar_color, -1)
    cv2.putText(frame, "buffer", (5, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    cv2.imshow("Sordo — Sign Language", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()