import os
import json
import base64
import numpy as np
from collections import deque
from rest_framework.decorators import api_view
from rest_framework.response import Response

# =========================
# BASE DIR & LABELS
# Only lightweight things at module level
# =========================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')

with open(LABELS_PATH, 'r') as f:
    sign_labels = json.load(f)

print(f"✅ Labels loaded | Words: {sign_labels}")

# =========================
# LAZY GLOBALS
# cv2, mediapipe, tensorflow all load on first request
# =========================
_lstm_model     = None
_hands_detector = None
_cv2            = None

def get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as cv2_module
        _cv2 = cv2_module
    return _cv2

def get_model():
    global _lstm_model
    if _lstm_model is None:
        os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
        from tensorflow.keras.models import load_model
        path        = os.path.join(BASE_DIR, 'models', 'sign_model.h5')
        _lstm_model = load_model(path)
        print("✅ LSTM model loaded")
    return _lstm_model

def get_hands():
    global _hands_detector
    if _hands_detector is None:
        os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
        import mediapipe as mp
        mp_hands        = mp.solutions.hands
        _hands_detector = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe hands loaded")
    return _hands_detector

# =========================
# SESSION STATE
# =========================
sessions = {}

SEQUENCE_LENGTH      = 30
PREDICTION_THRESHOLD = 0.60
BUFFER_SIZE          = 15
CONFIRM_VOTES        = 4
NO_HAND_RESET_AFTER  = 10

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            'sequence':      [],
            'pred_buffer':   deque(maxlen=BUFFER_SIZE),
            'no_hand_count': 0,
        }
    return sessions[session_id]

def clear_session(session_id):
    sessions[session_id] = {
        'sequence':      [],
        'pred_buffer':   deque(maxlen=BUFFER_SIZE),
        'no_hand_count': 0,
    }

# =========================
# NORMALIZATION — mirrors collect_data.py exactly
# =========================
def normalize_landmarks(raw_126):
    arr = np.array(raw_126, dtype=np.float32).reshape(2, 21, 3)
    for h in range(2):
        if np.any(arr[h] != 0):
            arr[h] -= arr[h, 0, :]
            scale    = np.max(np.abs(arr[h])) + 1e-6
            arr[h]  /= scale
    return arr.flatten().tolist()

# =========================
# HAND ASSIGNMENT — mirrors collect_data.py exactly
# slot 0 = right hand, slot 1 = left hand
# =========================
def assign_hands(result):
    landmarks  = [0.0] * 126
    handedness = result.multi_handedness if result.multi_handedness else []

    for idx, hand_lms in enumerate(result.multi_hand_landmarks[:2]):
        label = handedness[idx].classification[0].label if idx < len(handedness) else 'Right'
        slot  = 0 if label == 'Right' else 1

        for j, lm in enumerate(hand_lms.landmark):
            base              = slot * 63 + j * 3
            landmarks[base]   = lm.x
            landmarks[base+1] = lm.y
            landmarks[base+2] = lm.z

    return normalize_landmarks(landmarks)

# =========================
# ENDPOINT 1 — /api/frame/
# =========================
@api_view(['POST'])
def predict_frame(request):
    try:
        session_id = request.data.get('session_id', 'default')
        frame_b64  = request.data.get('frame', None)

        if not frame_b64:
            return Response({'error': 'No frame provided'}, status=400)

        session = get_session(session_id)
        cv2     = get_cv2()

        # Decode base64 image
        img_bytes = base64.b64decode(frame_b64)
        img_arr   = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return Response({'error': 'Could not decode frame'}, status=400)

        # Flip + convert — mirrors predict_realtime.py
        frame     = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe detection
        result = get_hands().process(frame_rgb)

        if result.multi_hand_landmarks:
            session['no_hand_count'] = 0

            frame_lms = assign_hands(result)
            session['sequence'].append(frame_lms)

            if len(session['sequence']) > SEQUENCE_LENGTH:
                session['sequence'].pop(0)

            # Predict when sequence is full
            if len(session['sequence']) == SEQUENCE_LENGTH:
                input_data     = np.expand_dims(
                    np.array(session['sequence'], dtype=np.float32), axis=0
                )
                probs          = get_model().predict(input_data, verbose=0)[0]
                predicted_idx  = int(np.argmax(probs))
                confidence     = float(probs[predicted_idx])
                predicted_word = sign_labels[predicted_idx]

                if confidence >= PREDICTION_THRESHOLD:
                    session['pred_buffer'].append(predicted_word)
                    votes = list(session['pred_buffer']).count(predicted_word)

                    if votes >= CONFIRM_VOTES:
                        session['pred_buffer'].clear()
                        return Response({
                            'detected':    True,
                            'word':        predicted_word,
                            'confidence':  round(confidence * 100, 1),
                            'hand':        True,
                            'buffer_fill': len(session['sequence']),
                        })
                else:
                    session['pred_buffer'].clear()

            return Response({
                'detected':    False,
                'word':        None,
                'confidence':  0,
                'hand':        True,
                'buffer_fill': len(session['sequence']),
            })

        else:
            session['no_hand_count'] += 1

            if session['no_hand_count'] >= NO_HAND_RESET_AFTER:
                clear_session(session_id)
                return Response({
                    'detected':    False,
                    'word':        None,
                    'confidence':  0,
                    'hand':        False,
                    'buffer_fill': 0,
                })

            return Response({
                'detected':    False,
                'word':        None,
                'confidence':  0,
                'hand':        False,
                'buffer_fill': len(session['sequence']),
            })

    except Exception as e:
        return Response({'error': str(e)}, status=500)


# =========================
# ENDPOINT 2 — /api/predict/ (backward compat)
# =========================
@api_view(['POST'])
def predict_sign(request):
    try:
        sequence = request.data.get('sequence', [])

        if not sequence:
            return Response({'error': 'No sequence provided'}, status=400)

        sequence = np.array(sequence)

        if sequence.shape != (30, 126):
            return Response({'error': f'Wrong shape: {sequence.shape}'}, status=400)

        input_data     = np.expand_dims(sequence, axis=0)
        prediction     = get_model().predict(input_data, verbose=0)[0]
        confidence     = float(np.max(prediction))
        predicted_idx  = int(np.argmax(prediction))
        predicted_word = sign_labels[predicted_idx]

        if confidence < 0.60:
            return Response({
                'detected':   False,
                'word':       None,
                'confidence': round(confidence * 100, 1)
            })

        return Response({
            'detected':   True,
            'word':       predicted_word,
            'confidence': round(confidence * 100, 1)
        })

    except Exception as e:
        return Response({'error': str(e)}, status=500)