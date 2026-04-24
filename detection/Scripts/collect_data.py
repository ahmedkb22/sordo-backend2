import cv2
import numpy as np
import mediapipe as mp
import os

# =========================
# CONFIG
# =========================
SAVE_PATH       = "../wlasl/processed"
SEQUENCE_LENGTH = 30
SAMPLES_TO_COLLECT = 30   # increased from 20

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,   # raised: fewer ghost detections
    min_tracking_confidence=0.5
)

# =========================
# LANDMARK NORMALIZATION
# Centres each hand on its own wrist and scales to [-1, 1]
# so position / distance from camera doesn't matter.
# =========================
def normalize_landmarks(raw_126):
    arr = np.array(raw_126, dtype=np.float32).reshape(2, 21, 3)
    for h in range(2):
        if np.any(arr[h] != 0):                    # hand is present
            arr[h] -= arr[h, 0, :]                 # centre on wrist
            scale = np.max(np.abs(arr[h])) + 1e-6
            arr[h] /= scale                        # normalise to [-1, 1]
    return arr.flatten().tolist()

# =========================
# HAND ASSIGNMENT (consistent left / right slot)
# slot 0 = right hand, slot 1 = left hand
# =========================
def assign_hands(result):
    landmarks = [0.0] * 126
    if not result.multi_hand_landmarks:
        return landmarks

    handedness = result.multi_handedness if result.multi_handedness else []

    for idx, hand_lms in enumerate(result.multi_hand_landmarks[:2]):
        # MediaPipe labels are mirrored for selfie-cam, so "Right" → slot 0
        label = handedness[idx].classification[0].label if idx < len(handedness) else "Right"
        slot  = 0 if label == "Right" else 1

        for j, lm in enumerate(hand_lms.landmark):
            base = slot * 63 + j * 3
            landmarks[base]     = lm.x
            landmarks[base + 1] = lm.y
            landmarks[base + 2] = lm.z

    return normalize_landmarks(landmarks)

# =========================
# COLLECT ONE WORD
# =========================
def collect_word(word, cap):
    save_dir = os.path.join(SAVE_PATH, word)
    os.makedirs(save_dir, exist_ok=True)

    existing_count = len([f for f in os.listdir(save_dir) if f.endswith(".npy")])
    sample_count   = 0

    print(f"\n🎯 Collecting: {word.upper()}  (target: {SAMPLES_TO_COLLECT} new samples)")
    print("   Press SPACE to start each sample — Q to quit\n")

    while sample_count < SAMPLES_TO_COLLECT:
        sequence  = []
        recording = False

        print(f"  📸 Sample {sample_count + 1}/{SAMPLES_TO_COLLECT} — press SPACE")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── process once per loop iteration ──────────────────────────
            frame     = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result    = hands.process(frame_rgb)

            # draw landmarks
            if result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks[:2]:
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # ── UI overlay ────────────────────────────────────────────────
            cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)
            if recording:
                progress = int((len(sequence) / SEQUENCE_LENGTH) * 580)
                cv2.rectangle(frame, (30, 55), (30 + progress, 70), (0, 0, 255), -1)
                cv2.putText(frame,
                            f"● REC  {len(sequence)}/{SEQUENCE_LENGTH}",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                hands_detected = "✓ hands" if result.multi_hand_landmarks else "no hands"
                cv2.putText(frame,
                            f"{word.upper()}  [{hands_detected}]  SPACE=rec  Q=quit",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.imshow("Collect Data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                return False                        # signal to stop entirely

            if key == ord(' ') and not recording:
                recording = True
                sequence  = []
                print("    → Recording …")

            # ── collect frame into sequence ───────────────────────────────
            if recording:
                frame_lms = assign_hands(result)
                sequence.append(frame_lms)

                if len(sequence) == SEQUENCE_LENGTH:
                    # reject if fewer than 5 frames had any hand visible
                    non_zero = sum(1 for f in sequence if any(v != 0 for v in f))
                    if non_zero < 5:
                        print("    ⚠  Too few hand detections — discarded, try again")
                    else:
                        idx       = existing_count + sample_count
                        save_path = os.path.join(save_dir, f"custom_{idx:04d}.npy")
                        np.save(save_path, np.array(sequence, dtype=np.float32))
                        print(f"    ✅ Saved  ({non_zero}/{SEQUENCE_LENGTH} frames with hands)")
                        sample_count += 1
                    break

    return True

# =========================
# MAIN
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("🎥  Data Collection Tool — 2-hand, normalised landmarks")

WORDS_TO_COLLECT = ["abdessamed", "ahmed"]

for word in WORDS_TO_COLLECT:
    ok = collect_word(word, cap)
    if not ok:
        print("🛑 Quit early.")
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Collection done!  Next step:")
print("   python extract_landmarks.py   (if using WLASL videos)")
print("   python train_model.py")