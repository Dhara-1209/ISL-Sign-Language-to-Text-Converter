import os
import sys
import time
import pickle
import urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.tts import speak  # noqa: E402

MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "isl_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")

# MediaPipe Hand Landmarker model
MP_TASK_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")
MP_TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

BUFFER_MAXLEN       = 15
COMMIT_THRESHOLD    = 12
CONFIDENCE_THRESH   = 0.80
COMMIT_COOLDOWN_SEC = 1.5

# Manual hand connections for drawing skeleton
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),          # Thumb
    (0,5), (5,6), (6,7), (7,8),          # Index
    (5,9), (9,10), (10,11), (11,12),     # Middle
    (9,13), (13,14), (14,15), (15,16),   # Ring
    (13,17), (17,18), (18,19), (19,20),  # Pinky
    (0,17)                               # Palm
]


def download_hand_model():
    """Download the MediaPipe model if it doesn't exist."""
    if not os.path.exists(MP_TASK_MODEL):
        print("[*] Downloading MediaPipe Hand Landmarker model...")
        urllib.request.urlretrieve(MP_TASK_URL, MP_TASK_MODEL)
        print("[*] Download complete.")


def load_artifacts():
    """Load the trained Keras model, label encoder, and scaler."""
    print("[*] Loading model artifacts …")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, encoder, scaler


def draw_hand_skeleton(frame, hand_landmarks):
    """Draw hand landmarks and connections manually using OpenCV."""
    h, w, _ = frame.shape
    points = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))
        
    # Draw connections
    for conn in HAND_CONNECTIONS:
        cv2.line(frame, points[conn[0]], points[conn[1]], (0, 0, 255), 2)
        
    # Draw points
    for pt in points:
        cv2.circle(frame, pt, 3, (0, 255, 0), -1)


def run_prediction_loop() -> None:
    """Main real-time prediction loop."""
    model, encoder, scaler = load_artifacts()
    download_hand_model()
    
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MP_TASK_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open camera.")
        sys.exit(1)

    buffer = deque(maxlen=BUFFER_MAXLEN)
    current_word = ""
    last_commit_time = 0.0
    prev_frame_time = time.time()
    fps = 0.0

    print("\n[*] Starting prediction loop … (Press Q to quit)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            dt = now - prev_frame_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_frame_time = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # Detect hands using video mode
            detection_result = landmarker.detect_for_video(mp_image, int(now * 1000))

            letter = "-"
            confidence = 0.0

            if detection_result.hand_landmarks:
                hand_lms = detection_result.hand_landmarks[0]

                wrist_x, wrist_y, wrist_z = hand_lms[0].x, hand_lms[0].y, hand_lms[0].z
                landmarks = []
                for lm in hand_lms:
                    landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
                landmarks = np.array(landmarks, dtype=np.float32).reshape(1, -1)

                scaled = scaler.transform(landmarks)
                probs = model.predict(scaled, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                letter = str(encoder.inverse_transform([pred_idx])[0])

                draw_hand_skeleton(frame, hand_lms)

                buffer.append(letter)
                if len(buffer) >= COMMIT_THRESHOLD:
                    count = sum(1 for l in buffer if l == letter)
                    if count >= COMMIT_THRESHOLD and confidence > CONFIDENCE_THRESH and (now - last_commit_time) > COMMIT_COOLDOWN_SEC:
                        current_word += letter
                        last_commit_time = now
                        buffer.clear()
                        print(f"    ✓ Committed '{letter}' → word: '{current_word}'")

            # Draw UI
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Letter: {letter}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.rectangle(frame, (0, h - 70), (w, h - 30), (0, 0, 0), -1)
            cv2.putText(frame, f"Word: {current_word}", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, "[SPACE] Speak  [BS] Del  [C] Clear  [Q] Quit", (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow("ISL to Text & Speech Converter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == 32:
                if current_word:
                    speak(current_word)
            elif key == 8 or key == 127:
                if current_word:
                    current_word = current_word[:-1]
            elif key == ord("c"):
                current_word = ""
                buffer.clear()

    except Exception as exc:
        print(f"\n✗ Error: {exc}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_prediction_loop()