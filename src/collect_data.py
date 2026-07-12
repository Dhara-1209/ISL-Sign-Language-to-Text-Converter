import os
import sys
import argparse
import urllib.request
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

NUM_LANDMARKS = 21
COORDS_PER_LM = 3
FEATURE_COUNT = NUM_LANDMARKS * COORDS_PER_LM

EXPECTED_CLASSES = (
    [str(i) for i in range(10)] +
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Path to save the MediaPipe tasks model
MODEL_DIR = os.path.join("models", "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def download_hand_model():
    """Download the MediaPipe Hand Landmarker model if it doesn't exist."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
        print("[*] Downloading MediaPipe Hand Landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_DIR)
        print("[*] Download complete.")


def extract_landmarks_from_image(image_bgr, landmarker):
    """Run MediaPipe Hand Landmarker on a single BGR image."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    detection_result = landmarker.detect(mp_image)

    if detection_result.hand_landmarks:
        hand_lms = detection_result.hand_landmarks[0]
        
        # Get wrist coordinates (Landmark 0) to normalize position
        wrist_x, wrist_y, wrist_z = hand_lms[0].x, hand_lms[0].y, hand_lms[0].z
        
        landmarks = []
        for lm in hand_lms:
            # Make coordinates relative to the wrist
            landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
            
        return np.array(landmarks, dtype=np.float32)

    return None

def process_dataset(dataset_root: str, output_csv: str) -> None:
    """Walk every sub-folder, extract landmarks, and save to CSV."""
    download_hand_model()
    
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_DIR),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    all_rows = []
    skipped_count = 0
    processed_count = 0

    class_folders = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ])

    print(f"\n{'='*60}")
    print(f"  ISL Landmark Extraction")
    print(f"  Classes found: {len(class_folders)} → {class_folders}")
    print(f"{'='*60}\n")

    for idx, class_name in enumerate(class_folders):
        class_dir = os.path.join(dataset_root, class_name)
        images = [
            f for f in os.listdir(class_dir)
            if os.path.splitext(f.lower())[1] in VALID_EXTENSIONS
        ]
        print(f"[{idx+1}/{len(class_folders)}] Processing class '{class_name}' — {len(images)} images …")

        class_skipped = 0
        class_success = 0

        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            try:
                image = cv2.imread(img_path)
                if image is None:
                    class_skipped += 1
                    skipped_count += 1
                    continue

                landmarks = extract_landmarks_from_image(image, landmarker)
                if landmarks is not None:
                    row = landmarks.tolist() + [class_name]
                    all_rows.append(row)
                    processed_count += 1
                    class_success += 1
                else:
                    class_skipped += 1
                    skipped_count += 1
            except Exception as exc:
                print(f"    ✗ Error on {img_path}: {exc}")
                class_skipped += 1
                skipped_count += 1

        print(f"    ✓ {class_success} extracted | ✗ {class_skipped} skipped")

    column_names = [
        f"{coord}{i}"
        for i in range(NUM_LANDMARKS)
        for coord in ("x", "y", "z")
    ] + ["label"]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(all_rows, columns=column_names)
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Total images processed : {processed_count}")
    print(f"  Total images skipped   : {skipped_count}")
    print(f"  CSV rows               : {len(df)}")
    print(f"  CSV saved to           : {output_csv}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand landmarks.")
    parser.add_argument("--dataset_path", type=str, default="data/raw")
    parser.add_argument("--output_path", type=str, default=os.path.join("data", "landmarks", "isl_landmarks.csv"))
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        print(f"✗ Dataset root not found: {args.dataset_path}")
        sys.exit(1)

    process_dataset(args.dataset_path, args.output_path)