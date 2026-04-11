import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# =========================
# CONFIG
# =========================
VIDEO_PATH = "../wlasl/videos"
OUTPUT_PATH = "../wlasl/processed"
ANNOTATION_PATH = "../wlasl/WLASL_v0.3.json"
SEQUENCE_LENGTH = 30

TARGET_WORDS = ["hello","yes", "no", "help","my","name"]

# =========================
# MEDIAPIPE INIT
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,          # ✅ 2 hands
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# =========================
# LOAD ANNOTATIONS
# =========================
with open(ANNOTATION_PATH, "r") as f:
    data = json.load(f)

# =========================
# FUNCTION: EXTRACT LANDMARKS
# =========================
def extract_from_video(video_file):
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_file}")
        return None

    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            # ✅ Always 126 values (2 hands x 21 landmarks x 3 coords)
            landmarks = [0.0] * 126

            if result.multi_hand_landmarks:
                for i, hand in enumerate(result.multi_hand_landmarks[:2]):
                    for j, lm in enumerate(hand.landmark):
                        landmarks[i * 63 + j * 3]     = lm.x
                        landmarks[i * 63 + j * 3 + 1] = lm.y
                        landmarks[i * 63 + j * 3 + 2] = lm.z

            sequence.append(landmarks)

        except:
            sequence.append([0.0] * 126)

    cap.release()

    if len(sequence) == 0:
        return None

    # Normalize sequence length
    if len(sequence) < SEQUENCE_LENGTH:
        padding = [[0.0] * 126] * (SEQUENCE_LENGTH - len(sequence))
        sequence.extend(padding)
    else:
        sequence = sequence[:SEQUENCE_LENGTH]

    return np.array(sequence)  # shape: (30, 126)

# =========================
# MAIN PROCESS
# =========================
print("🚀 Starting extraction (2-hand mode)...")

for item in tqdm(data):
    word = item["gloss"]

    if word not in TARGET_WORDS:
        continue

    for instance in item["instances"]:
        video_id = instance["video_id"]
        video_file = os.path.join(VIDEO_PATH, video_id + ".mp4")

        if not os.path.exists(video_file):
            continue

        sequence = extract_from_video(video_file)

        if sequence is None:
            continue

        save_dir = os.path.join(OUTPUT_PATH, word)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, video_id + ".npy")
        np.save(save_path, sequence)

print("✅ Extraction finished! Shape per file: (30, 126)")