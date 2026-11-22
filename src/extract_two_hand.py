"""
Extract RAW hand landmarks (up to 2 hands) from every dataset image.

Saves data/landmarks/raw_hands.npz with:
    hands   : (N, 2, 21, 3) float32  raw MediaPipe coords, NaN where no hand
    n_hands : (N,) int8              1 or 2
    labels  : (N,) str               class folder name
    index   : (N,) int32             numeric part of the file name (1..1000)

Run from the project root:
    python src/extract_two_hand.py
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(PROJECT_ROOT, "dataset")
TASK = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
OUT = os.path.join(PROJECT_ROOT, "data", "landmarks", "raw_hands.npz")

_lm = None


def _init():
    global _lm
    opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=TASK),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    _lm = vision.HandLandmarker.create_from_options(opts)


def _work(item):
    label, fname = item
    img = cv2.imread(os.path.join(DATASET, label, fname))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = _lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None
    hands = np.full((2, 21, 3), np.nan, dtype=np.float32)
    n = min(len(res.hand_landmarks), 2)
    for i in range(n):
        hands[i] = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[i]],
                            dtype=np.float32)
    stem = os.path.splitext(fname)[0]
    idx = int(stem) if stem.isdigit() else -1
    return hands, n, label, idx


def main():
    items = []
    for label in sorted(os.listdir(DATASET)):
        d = os.path.join(DATASET, label)
        if os.path.isdir(d):
            items += [(label, f) for f in sorted(os.listdir(d))]
    print(f"{len(items)} images", flush=True)

    hands, nh, labels, idxs = [], [], [], []
    with Pool(os.cpu_count() or 2, initializer=_init) as pool:
        for k, r in enumerate(pool.imap(_work, items, chunksize=50)):
            if r is not None:
                hands.append(r[0]); nh.append(r[1]); labels.append(r[2]); idxs.append(r[3])
            if k % 2000 == 0:
                print(f"  {k}/{len(items)}  kept {len(hands)}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, hands=np.array(hands), n_hands=np.array(nh, dtype=np.int8),
                        labels=np.array(labels), index=np.array(idxs, dtype=np.int32))
    print(f"saved {len(hands)} samples -> {OUT}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
