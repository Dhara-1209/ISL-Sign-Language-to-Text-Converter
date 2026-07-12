"""
Download MediaPipe hand landmarker model
"""
import os
from pathlib import Path
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MediaPipe hand landmarker model URL (official)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task"
MODEL_PATH = Path("models/hand_landmarker.task")

def download_model():
    """Download the hand landmarker model."""
    # Create models directory
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    if MODEL_PATH.exists():
        logger.info(f"Model already exists at {MODEL_PATH}")
        return str(MODEL_PATH)
    
    logger.info(f"Downloading hand landmarker model from {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info(f"Model downloaded successfully to {MODEL_PATH}")
        return str(MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.info("Trying alternative approach...")
        return None

if __name__ == "__main__":
    model_path = download_model()
    if model_path:
        print(f"SUCCESS: Model ready at {model_path}")
    else:
        print("FAILED: Could not download model")
