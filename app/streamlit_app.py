import os
import sys
import time
import pickle
import urllib.request
from collections import deque

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.tts import speak  # noqa: E402

MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "isl_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
MP_TASK_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")
MP_TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

BUFFER_MAXLEN       = 15
COMMIT_THRESHOLD    = 12
CONFIDENCE_THRESH   = 0.80
COMMIT_COOLDOWN_SEC = 1.5

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20), (0,17)
]

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ── Shared State Object ──────────────────────────────────────
class AppState:
    def __init__(self):
        self.letter = "-"
        self.confidence = 0.0
        self.word = ""
        self.clear_flag = False
        self.delete_flag = False

if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()

app_state = st.session_state.app_state

@st.cache_resource
def download_hand_model():
    if not os.path.exists(MP_TASK_MODEL):
        urllib.request.urlretrieve(MP_TASK_URL, MP_TASK_MODEL)

@st.cache_resource
def load_model_artifacts():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, encoder, scaler
    except Exception as exc:
        st.error(f"⚠ Could not load model artifacts: {exc}")
        return None

class ISLVideoTransformer(VideoTransformerBase):
    def __init__(self):
        download_hand_model()
        artifacts = load_model_artifacts()
        if artifacts is not None:
            self.model, self.encoder, self.scaler = artifacts
        else:
            self.model = self.encoder = self.scaler = None

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MP_TASK_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.buffer = deque(maxlen=BUFFER_MAXLEN)
        self.last_commit_time = 0.0
        self.state = app_state
        self.frame_count = 0  # For frame skipping

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Handle UI Commands immediately
        if self.state.clear_flag:
            self.state.word = ""
            self.buffer.clear()
            self.state.clear_flag = False
            
        if self.state.delete_flag:
            if self.state.word:
                self.state.word = self.state.word[:-1]
            self.state.delete_flag = False

        if self.model is None:
            cv2.putText(img, "Model not loaded!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Frame skipping for performance (process every 2nd frame)
        self.frame_count += 1
        if self.frame_count % 2 == 0:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            timestamp_ms = int(time.time() * 1000)
            detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if detection_result.hand_landmarks:
                hand_lms = detection_result.hand_landmarks[0]

                # Normalization Fix
                wrist_x, wrist_y, wrist_z = hand_lms[0].x, hand_lms[0].y, hand_lms[0].z
                landmarks = []
                for lm in hand_lms:
                    landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
                landmarks = np.array(landmarks, dtype=np.float32).reshape(1, -1)

                try:
                    scaled = self.scaler.transform(landmarks)
                    probs = self.model.predict(scaled, verbose=0)[0]
                    pred_idx = int(np.argmax(probs))
                    self.state.confidence = float(probs[pred_idx])
                    self.state.letter = str(self.encoder.inverse_transform([pred_idx])[0])
                except Exception:
                    self.state.letter = "-"
                    self.state.confidence = 0.0

                self.buffer.append(self.state.letter)
                now = time.time()
                if len(self.buffer) >= COMMIT_THRESHOLD:
                    count = sum(1 for l in self.buffer if l == self.state.letter)
                    if count >= COMMIT_THRESHOLD and self.state.confidence > CONFIDENCE_THRESH and (now - self.last_commit_time) > COMMIT_COOLDOWN_SEC:
                        self.state.word += self.state.letter
                        self.last_commit_time = now
                        self.buffer.clear()
            else:
                self.state.letter = "-"
                self.state.confidence = 0.0

        # Draw skeleton and UI overlay on EVERY frame
        if detection_result.hand_landmarks:
            hand_lms = detection_result.hand_landmarks[0]
            h, w, _ = img.shape
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for conn in HAND_CONNECTIONS:
                cv2.line(img, points[conn[0]], points[conn[1]], (0, 0, 255), 2)
            for pt in points:
                cv2.circle(img, pt, 3, (0, 255, 0), -1)

        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(img, f"Letter: {self.state.letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, f"Conf: {self.state.confidence*100:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.rectangle(img, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.putText(img, f"Word: {self.state.word}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Page Configuration ───────────────────────────────────────
st.set_page_config(page_title="ISL to Text & Speech Converter", page_icon="🤟", layout="wide")

with st.sidebar:
    st.markdown("## 🤟 ISL Converter")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | **Architecture** | MLP (63→128→64→36) |
    | **Classes** | 36 (0-9, A-Z) |
    | **Accuracy** | 98.27% |
    """)
    st.markdown("### 📝 Instructions")
    st.markdown("1. Click **START** to enable webcam\n2. Show ISL gestures\n3. Use buttons below to control text")

st.title("🤟 Indian Sign Language → Text & Speech Converter")
st.markdown("---")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📹 Live Webcam Feed")
    ctx = webrtc_streamer(
        key="isl-stream",
        video_transformer_factory=ISLVideoTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_right:
    st.subheader("🎯 Controls")
    state = app_state

    if ctx and ctx.state.playing:
        st.markdown("#### ✍️ Current Word")
        st.markdown(f"<h2 style='text-align: center; color: #00FF00; background-color: #1e1e2e; padding: 15px; border-radius: 10px;'>{state.word}</h2>", unsafe_allow_html=True)
        
        st.markdown(f"**Predicted Letter:** `{state.letter}`")
        st.markdown(f"**Confidence:** `{state.confidence*100:.1f}%`")
        st.progress(min(state.confidence, 1.0))

        st.markdown("---")
        b1, b2, b3 = st.columns(3)
        
        with b1:
            if st.button("🔊 Speak", use_container_width=True):
                if state.word:
                    with st.spinner("Speaking…"):
                        speak(state.word)
                        
        with b2:
            if st.button("⬅️ Delete", use_container_width=True):
                if state.word:
                    state.word = state.word[:-1]
                state.delete_flag = True
                
        with b3:
            if st.button("🗑️ Clear", use_container_width=True):
                state.word = ""
                state.clear_flag = True
    else:
        st.info("👈 Click **START** in the webcam panel to begin.")