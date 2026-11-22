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
from src.features import build_features  # noqa: E402

MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "isl_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
MP_TASK_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")
MP_TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# ── Prediction parameters ─────────────────────────────────────
# Consecutive-frame strategy:  the SAME letter must appear for
# CONSECUTIVE_REQUIRED frames in a row (all at or above CONFIDENCE_THRESH)
# before it is committed to the word.  Any frame that shows a different
# letter or low confidence resets the counter to zero, so transitional
# frames between gestures can NEVER trigger a commit.
CONSECUTIVE_REQUIRED    = 20     # frames of the same letter needed to commit
CONFIDENCE_THRESH       = 0.70   # minimum per-frame confidence to count
COMMIT_COOLDOWN_SEC     = 2.0    # minimum gap between two commits
NO_HAND_FRAMES_FOR_SPACE = 22    # consecutive no-hand frames → auto-space

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20), (0,17)
]

def _read_model_accuracy() -> str:
    """Read the accuracy written by train_two_hand.py (no hard-coded numbers)."""
    import json
    try:
        with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
            return f"{json.load(f)['block_split_accuracy'] * 100:.1f}%"
    except Exception:
        return "n/a"


MODEL_ACC = _read_model_accuracy()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ── Shared State Object ──────────────────────────────────────
class AppState:
    def __init__(self):
        self.letter = "-"
        self.confidence = 0.0
        self.word = ""
        self.sentence = ""         # Accumulated full sentence
        self.clear_flag = False
        self.delete_flag = False
        self.space_flag = False
        self.speak_flag = False

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
            num_hands=2,                    # many ISL letters use BOTH hands
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_commit_time = 0.0
        self.no_hand_frame_count = 0
        self.space_added = False
        self.state = app_state
        self._last_detection = None

        # ── Consecutive-frame tracking ────────────────────────
        # Only commit a letter when the SAME letter appears
        # CONSECUTIVE_REQUIRED frames IN A ROW at high confidence.
        # Any different letter or low-confidence frame resets to 0.
        self.consec_letter = None   # letter currently being accumulated
        self.consec_count  = 0      # how many consecutive frames so far

    # ── Image enhancement ──────────────────────────────────────
    @staticmethod
    def _enhance_frame(img: np.ndarray) -> np.ndarray:
        """Sharpen + contrast-boost the frame for better landmark detection."""
        # Mild unsharp mask
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        # CLAHE on the luminance channel
        lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return enhanced

    # ── Draw hand skeleton ─────────────────────────────────────
    @staticmethod
    def _draw_hand(img, hand_lms):
        h, w, _ = img.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        # Draw connections
        for conn in HAND_CONNECTIONS:
            cv2.line(img, points[conn[0]], points[conn[1]], (0, 200, 255), 2)
        # Draw joints
        for i, pt in enumerate(points):
            radius = 5 if i == 0 else 4   # Wrist slightly bigger
            color = (0, 255, 100) if i == 0 else (255, 255, 255)
            cv2.circle(img, pt, radius, color, -1)
        # Bounding box around hand
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        pad = 20
        x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
        x2, y2 = min(w, max(xs) + pad), min(h, max(ys) + pad)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 1)

    # ── Draw HUD overlay ──────────────────────────────────────
    @staticmethod
    def _draw_hud(img, letter, confidence, word, consec_count, consec_required):
        h, w = img.shape[:2]

        # Top bar (letter + confidence)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (15, 15, 30), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        conf_pct = confidence * 100
        bar_color = (0, 220, 80) if conf_pct >= 82 else (0, 180, 255) if conf_pct >= 60 else (60, 60, 200)

        cv2.putText(img, f"Letter: {letter}", (12, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Conf: {conf_pct:.1f}%", (12, 62),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, bar_color, 1, cv2.LINE_AA)

        # Confidence bar (right side of top bar)
        bx1, bx2 = 170, w - 10
        cv2.rectangle(img, (bx1, 48), (bx2, 64), (50, 50, 70), -1)
        cbar = int((bx2 - bx1) * min(confidence, 1.0))
        cv2.rectangle(img, (bx1, 48), (bx1 + cbar, 64), bar_color, -1)

        # Commit-progress bar: how close to committing the current letter
        # Shown BELOW the confidence bar in orange/green
        progress = min(consec_count / max(consec_required, 1), 1.0)
        prog_color = (0, 220, 80) if progress >= 1.0 else (0, 165, 255)
        cv2.rectangle(img, (bx1, 66), (bx2, 78), (30, 30, 50), -1)
        pbar = int((bx2 - bx1) * progress)
        if pbar > 0:
            cv2.rectangle(img, (bx1, 66), (bx1 + pbar, 78), prog_color, -1)
        cv2.putText(img, f"Hold: {consec_count}/{consec_required}", (bx1, 62),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (180, 180, 180), 1, cv2.LINE_AA)

        # Bottom bar (committed word)
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (0, h - 60), (w, h), (15, 15, 30), -1)
        cv2.addWeighted(overlay2, 0.75, img, 0.25, 0, img)
        display_word = word if word else "_"
        cv2.putText(img, f"Word: {display_word}", (12, h - 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 220), 1, cv2.LINE_AA)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # ── Handle UI commands ───────────────────────────────
        if self.state.clear_flag:
            self.state.word = ""
            self.state.sentence = ""
            self.consec_letter = None
            self.consec_count  = 0
            self.state.clear_flag = False

        if self.state.delete_flag:
            if self.state.word:
                self.state.word = self.state.word[:-1]
            self.state.delete_flag = False

        if self.state.space_flag:
            if self.state.word and not self.state.word.endswith(" "):
                self.state.word += " "
            self.state.space_flag = False
            self.space_added = True

        # NOTE: no sharpen/CLAHE here. The model was trained on raw images,
        # so enhancing the live frame only shifts the landmarks away from
        # what the model learned.
        if self.model is None:
            cv2.putText(img, "Model not loaded!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # ── Run detection on every frame ─────────────────────
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        self._last_detection = detection_result

        if detection_result.hand_landmarks:
            self.no_hand_frame_count = 0
            self.space_added = False
            hand_lms = detection_result.hand_landmarks[0]

            # ── Wrist-relative normalisation (must match training) ───
            # Training CSV has wrist zeroed: x0=0, y0=0, z0=0 and all
            # other landmarks are (lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z).
            # Do NOT apply any further scaling — the scaler.pkl handles that.
            hands_raw = [
                np.array([[p.x, p.y, p.z] for p in h], dtype=np.float32)
                for h in detection_result.hand_landmarks[:2]
            ]
            # Same feature builder as training (src/features.py): up to two
            # hands, size-normalised, ordered left-to-right.
            landmarks = build_features(hands_raw).reshape(1, -1)

            try:
                scaled = self.scaler.transform(landmarks)
                probs  = self.model.predict(scaled, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                raw_confidence = float(probs[pred_idx])
                raw_letter     = str(self.encoder.inverse_transform([pred_idx])[0])
            
            except Exception as exc:
                import traceback
                print("PREDICT ERROR:", repr(exc), flush=True)
                traceback.print_exc()
                raw_letter     = "-"
                raw_confidence = 0.0

            self.state.letter     = raw_letter
            self.state.confidence = raw_confidence

            # ── Consecutive-frame commit logic ────────────────
            # Rule: letter must be the SAME and confidence >= threshold
            # for CONSECUTIVE_REQUIRED frames in a row.
            # Any deviation resets the counter to zero immediately.
            now = time.time()
            if raw_confidence >= CONFIDENCE_THRESH and raw_letter != "-":
                if raw_letter == self.consec_letter:
                    self.consec_count += 1
                else:
                    # Different letter → start fresh streak
                    self.consec_letter = raw_letter
                    self.consec_count  = 1
            else:
                # Low confidence or no prediction → break streak
                self.consec_letter = None
                self.consec_count  = 0

            # Commit when streak is long enough AND cooldown has passed
            if (
                self.consec_count >= CONSECUTIVE_REQUIRED
                and (now - self.last_commit_time) > COMMIT_COOLDOWN_SEC
            ):
                self.state.word      += self.consec_letter
                self.last_commit_time = now
                # Reset streak so next letter starts fresh
                self.consec_letter = None
                self.consec_count  = 0

            # Draw hand skeleton(s) on original img
            for h in detection_result.hand_landmarks[:2]:
                self._draw_hand(img, h)

        else:
            # ── No hand in frame ─────────────────────────────
            self.no_hand_frame_count += 1
            self.state.letter     = "-"
            self.state.confidence = 0.0
            # Reset streak — user lifted hand to transition
            self.consec_letter = None
            self.consec_count  = 0

            # Auto-space after N consecutive no-hand frames
            if (
                self.no_hand_frame_count == NO_HAND_FRAMES_FOR_SPACE
                and self.state.word
                and not self.state.word.endswith(" ")
                and not self.space_added
            ):
                self.state.word  += " "
                self.space_added  = True

        # ── Draw HUD ─────────────────────────────────────────
        self._draw_hud(
            img,
            self.state.letter,
            self.state.confidence,
            self.state.word,
            self.consec_count,
            CONSECUTIVE_REQUIRED,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="ISL to Text & Speech Converter",
    page_icon="🤟",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .word-display {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #00FFD0;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #00FFD066;
        min-height: 70px;
        letter-spacing: 4px;
    }
    .letter-badge {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FFD700;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🤟 ISL Converter")
    st.markdown("### 📊 Model Info")
    st.markdown(f"""
    | Property | Value |
    |----------|-------|
    | **Architecture** | MLP (129→256→128→classes) |
    | **Classes** | 35 (1–9, A–Z; 0 = O) to 34 (1, 3–9, A–Z; 0 = O, 2 = V)|
    | **Held-out accuracy** | {MODEL_ACC} |
    | **Confidence threshold** | {int(CONFIDENCE_THRESH*100)}% |
    | **Consecutive frames** | {CONSECUTIVE_REQUIRED} |
    | **Cooldown** | {COMMIT_COOLDOWN_SEC}s |
    """)
    st.markdown("### 📝 How to use")
    st.markdown(
        "1. Click **START** to enable webcam\n"
        "2. Show an ISL hand sign steadily\n"
        "3. Watch the **orange hold bar** on the video fill up\n"
        "4. Letter commits when the bar is full ✅\n"
        "5. **Remove hand** briefly → auto-space between words\n"
        "6. Use **Space / Delete / Clear** buttons as needed\n"
        "7. Click **🔊 Speak** to hear the word"
    )
    st.markdown("---")
    st.markdown("#### ⚙️ Prediction Settings")
    st.caption(f"Consecutive frames required: {CONSECUTIVE_REQUIRED} | Confidence: {int(CONFIDENCE_THRESH*100)}% | Cooldown: {COMMIT_COOLDOWN_SEC}s")

st.title("🤟 Indian Sign Language → Text & Speech Converter")
st.markdown("---")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📹 Live Webcam Feed")
    ctx = webrtc_streamer(
        key="isl-stream",
        video_transformer_factory=ISLVideoTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width":  {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30},
                "facingMode": "user",
            },
            "audio": False,
        },
        async_processing=True,
    )

with col_right:
    st.subheader("🎯 Controls & Output")
    state = app_state

    if ctx and ctx.state.playing:
        # Current predicted letter
        st.markdown("#### 🔤 Predicted Letter")
        letter_display = state.letter if state.letter != "-" else "—"
        st.markdown(f"<div class='letter-badge'>{letter_display}</div>", unsafe_allow_html=True)
        st.progress(min(state.confidence, 1.0), text=f"Confidence: {state.confidence*100:.1f}%")

        st.markdown("---")

        # Word being built
        st.markdown("#### ✍️ Current Word / Text")
        word_display = state.word.strip() if state.word.strip() else "&nbsp;"
        st.markdown(f"<div class='word-display'>{word_display}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Action buttons
        b1, b2, b3, b4 = st.columns(4)

        with b1:
            if st.button("🔊 Speak", use_container_width=True):
                text = state.word.strip()
                if text:
                    with st.spinner("Speaking…"):
                        speak(text)

        with b2:
            if st.button("␣ Space", use_container_width=True):
                state.space_flag = True

        with b3:
            if st.button("⬅️ Delete", use_container_width=True):
                if state.word:
                    state.word = state.word[:-1]
                state.delete_flag = True

        with b4:
            if st.button("🗑️ Clear", use_container_width=True):
                state.word = ""
                state.clear_flag = True

        # Tips
        st.markdown("---")
        st.info(
            "💡 **Tip:** Watch the **orange 'Hold' bar** on the camera feed — "
            "keep your hand sign steady until it fills completely. "
            "Remove your hand briefly to insert a space."
        )

    else:
        st.info("👈 Click **START** in the webcam panel to begin.")
        st.markdown("""
        #### 📌 Quick Guide
        - Hold each sign **steadily** for ~1–2 seconds
        - Watch the orange **Hold bar** on the video fill up
        - Letter is committed only when the bar is **completely full**
        - Ensure **good lighting** on your hand
        - Keep hand **fully visible** in the frame
        - Remove hand to **add a space** automatically
        """)
