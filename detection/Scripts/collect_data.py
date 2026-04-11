import cv2
import numpy as np
import mediapipe as mp
import os

# =========================
# CONFIG
# =========================
SAVE_PATH = "../wlasl/processed"
SEQUENCE_LENGTH = 30
SAMPLES_TO_COLLECT = 20

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,              # ✅ 2 hands
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture(0)

def collect_word(word):
    save_dir = os.path.join(SAVE_PATH, word)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🎯 Get ready to sign: {word.upper()}")
    print("Press SPACE to start recording each sample")

    sample_count = 0

    while sample_count < SAMPLES_TO_COLLECT:
        sequence = []
        recording = False

        print(f"\n📸 Sample {sample_count + 1}/{SAMPLES_TO_COLLECT} — Press SPACE to record")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks[:2]:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # UI
            cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)

            if recording:
                cv2.putText(frame, f"RECORDING... {len(sequence)}/{SEQUENCE_LENGTH}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"SIGN: {word.upper()} | SPACE=Record Q=Quit",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Collect Data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

            if key == ord(' ') and not recording:
                recording = True
                sequence = []

            if recording:
                # ✅ 2 hands = 126 values
                landmarks = [0.0] * 126

                ret2, frame2 = cap.read()
                if ret2:
                    frame2 = cv2.flip(frame2, 1)
                    frame_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    result2 = hands.process(frame_rgb2)

                    if result2.multi_hand_landmarks:
                        for i, hand in enumerate(result2.multi_hand_landmarks[:2]):
                            for j, lm in enumerate(hand.landmark):
                                landmarks[i * 63 + j * 3]     = lm.x
                                landmarks[i * 63 + j * 3 + 1] = lm.y
                                landmarks[i * 63 + j * 3 + 2] = lm.z

                sequence.append(landmarks)

                if len(sequence) == SEQUENCE_LENGTH:
                    existing = len(os.listdir(save_dir))
                    save_path = os.path.join(save_dir, f"custom_{existing}.npy")
                    np.save(save_path, np.array(sequence))
                    print(f"  ✅ Saved sample {sample_count + 1}")
                    sample_count += 1
                    break

# =========================
# MAIN
# =========================
print("🎥 Data Collection Tool — 2 hands supported")

collect_word("help")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Collection done! Now retrain:")
print("   python train_model.py")