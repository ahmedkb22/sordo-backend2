import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import json
import collections

# =========================
# CONFIG
# =========================
MODEL_PATH = "../models/sign_model.h5"
LABELS_PATH = "../models/labels.json"
SEQUENCE_LENGTH = 30
PREDICTION_THRESHOLD = 0.60  # ✅ 60% minimum

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
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# =========================
# WEBCAM LOOP
# =========================
cap = cv2.VideoCapture(0)
sequence = []
predictions_buffer = collections.deque(maxlen=5)
current_word = ""
current_confidence = 0.0
frame_counter = 0

print("🎥 Webcam started — show a sign!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # ✅ 2 hands = 126 values
    landmarks = [0.0] * 126

    if result.multi_hand_landmarks:
        for i, hand in enumerate(result.multi_hand_landmarks[:2]):
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            for j, lm in enumerate(hand.landmark):
                landmarks[i * 63 + j * 3]     = lm.x
                landmarks[i * 63 + j * 3 + 1] = lm.y
                landmarks[i * 63 + j * 3 + 2] = lm.z

        sequence.append(landmarks)
        sequence = sequence[-SEQUENCE_LENGTH:]

    else:
        # ✅ Reset when hand disappears
        sequence = []
        current_word = ""
        current_confidence = 0.0
        predictions_buffer.clear()
        frame_counter = 0

    # ✅ Predict every 5 frames
    if len(sequence) == SEQUENCE_LENGTH and frame_counter % 5 == 0:
        input_data = np.expand_dims(sequence, axis=0)
        prediction = model.predict(input_data, verbose=0)[0]
        confidence = np.max(prediction)
        predicted_idx = np.argmax(prediction)

        if confidence >= PREDICTION_THRESHOLD:
            predicted_word = labels[predicted_idx]
            predictions_buffer.append(predicted_word)

            if predictions_buffer.count(predicted_word) >= 3:
                current_word = predicted_word
                current_confidence = confidence
        else:
            # ✅ Below 60% — not confident enough
            current_word = ""
            current_confidence = confidence
            predictions_buffer.clear()

    # =========================
    # DISPLAY
    # =========================
    h, w, _ = frame.shape

    cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)

    if current_word:
        # ✅ Show word + confidence percentage
        text = f"SIGN: {current_word.upper()}  ({int(current_confidence * 100)}%)"
        cv2.putText(frame, text,
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 100), 3)
    elif result.multi_hand_landmarks:
        # Hand visible but not confident
        if current_confidence > 0:
            text = f"No sign detected  ({int(current_confidence * 100)}%)"
        else:
            text = "Analyzing..."
        cv2.putText(frame, text,
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 100, 255), 2)
    else:
        # No hand at all
        cv2.putText(frame, "Show a sign...",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (100, 100, 100), 2)

    cv2.imshow("Sordo - Sign Language", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()