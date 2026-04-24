import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# =========================
# CONFIG
# =========================
VIDEO_PATH       = "../wlasl/videos"
OUTPUT_PATH      = "../wlasl/processed"
ANNOTATION_PATH  = "../wlasl/WLASL_v0.3.json"
SEQUENCE_LENGTH  = 30

TARGET_WORDS = ["hello", "yes", "no", "help", "my", "name", "abdessamed", "ahmed"]

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# =========================
# NORMALISATION  (same as collect_data.py — MUST stay in sync)
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
# CONSISTENT HAND ASSIGNMENT
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
# SMART FRAME SAMPLING
# Samples SEQUENCE_LENGTH frames spread evenly across the whole video
# instead of only taking the first N frames — captures the full gesture.
# =========================
def sample_frames_evenly(total_frames, n):
    if total_frames <= n:
        return list(range(total_frames))
    step = total_frames / n
    return [int(i * step) for i in range(n)]

# =========================
# EXTRACT FROM ONE VIDEO
# =========================
def extract_from_video(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return None

    # Read all frames first (videos are short — <5 s)
    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    if len(raw_frames) == 0:
        return None

    # Pick SEQUENCE_LENGTH frames spread across the whole video
    indices  = sample_frames_evenly(len(raw_frames), SEQUENCE_LENGTH)
    sequence = []

    for idx in indices:
        frame = raw_frames[idx]
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result    = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                landmarks = assign_hands(result)
            else:
                landmarks = [0.0] * 126

            sequence.append(landmarks)
        except Exception:
            sequence.append([0.0] * 126)

    # Pad if we got fewer frames than expected
    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append([0.0] * 126)

    arr = np.array(sequence, dtype=np.float32)  # (30, 126)

    # Quality check: skip videos where fewer than 8 frames have hands
    non_zero = np.sum(np.any(arr != 0, axis=1))
    if non_zero < 8:
        return None

    return arr

# =========================
# LOAD ANNOTATIONS
# =========================
with open(ANNOTATION_PATH, "r") as f:
    data = json.load(f)

# =========================
# MAIN PROCESS
# =========================
print("🚀 Starting extraction — even sampling + normalisation + hand assignment")
print(f"   Target words: {TARGET_WORDS}\n")

skipped = 0
saved   = 0

for item in tqdm(data):
    word = item["gloss"]
    if word not in TARGET_WORDS:
        continue

    for instance in item["instances"]:
        video_id   = instance["video_id"]
        video_file = os.path.join(VIDEO_PATH, video_id + ".mp4")

        if not os.path.exists(video_file):
            skipped += 1
            continue

        sequence = extract_from_video(video_file)
        if sequence is None:
            skipped += 1
            continue

        save_dir = os.path.join(OUTPUT_PATH, word)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, video_id + ".npy")
        np.save(save_path, sequence)
        saved += 1

print(f"\n✅ Done — saved: {saved} | skipped: {skipped}")
print(f"   Each file shape: ({SEQUENCE_LENGTH}, 126)")

# Print per-word counts
print("\n📊 Samples per word:")
for word in TARGET_WORDS:
    d = os.path.join(OUTPUT_PATH, word)
    if os.path.isdir(d):
        count = len([f for f in os.listdir(d) if f.endswith(".npy")])
        status = "✅" if count >= 20 else "⚠️ "
        print(f"   {status} {word}: {count}")