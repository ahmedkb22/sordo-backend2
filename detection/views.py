from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy as np
import os
import json
from tensorflow.keras.models import load_model

# =========================
# BASE DIR
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# LOAD LSTM MODEL
# =========================
LSTM_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sign_model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')

lstm_model = load_model(LSTM_MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    sign_labels = json.load(f)

print(f"✅ LSTM model loaded | Words: {sign_labels}")

# =========================
# ENDPOINT — custom sign language (LSTM)
# =========================
@api_view(['POST'])
def predict_sign(request):
    try:
        # Frontend sends 30 frames of landmarks (30 x 126)
        sequence = request.data.get('sequence', [])

        if not sequence:
            return Response({'error': 'No sequence provided'}, status=400)

        sequence = np.array(sequence)

        if sequence.shape != (30, 126):
            return Response({'error': f'Wrong shape: {sequence.shape}'}, status=400)

        input_data = np.expand_dims(sequence, axis=0)
        prediction = lstm_model.predict(input_data, verbose=0)[0]

        confidence = float(np.max(prediction))
        predicted_idx = int(np.argmax(prediction))
        predicted_word = sign_labels[predicted_idx]

        if confidence < 0.60:
            return Response({
                'detected': False,
                'word': None,
                'confidence': round(confidence * 100, 1)
            })

        return Response({
            'detected': True,
            'word': predicted_word,
            'confidence': round(confidence * 100, 1)
        })

    except Exception as e:
        return Response({'error': str(e)}, status=500)