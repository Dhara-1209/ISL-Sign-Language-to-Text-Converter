"""
Shared feature builder for ISL recognition (training AND live prediction).

WHY THIS FILE EXISTS
--------------------
Many ISL letters (A, D, E, F, H, K, M, N, S, T ...) use TWO hands. The old
pipeline kept only the first detected hand, so those letters looked
identical or random to the model. It also never normalised hand size, so
the same sign looked different at different distances from the camera.

Using ONE function for both training and inference guarantees that the
model sees exactly the same kind of numbers in both places.

Feature layout (129 values)
---------------------------
[  0: 63]  hand A  - 21 landmarks (x, y, z), wrist-relative, size-normalised
[ 63:126]  hand B  - same, or all zeros if only one hand is visible
[126:129]  offset of hand B's wrist from hand A's wrist (x, y, z), same scale

Hands are ordered left-to-right in the image (smaller wrist x first), so the
order does not depend on MediaPipe's left/right guess.
"""
from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
HAND_FEATS = NUM_LANDMARKS * 3          # 63
FEATURE_COUNT = HAND_FEATS * 2 + 3      # 129


def _hand_scale(pts: np.ndarray) -> float:
    """Size of a hand = distance wrist -> middle-finger knuckle (2D)."""
    d = np.linalg.norm(pts[9, :2] - pts[0, :2])
    return float(d) if d > 1e-6 else 1.0


def build_features(hands: list[np.ndarray]) -> np.ndarray:
    """
    Parameters
    ----------
    hands : list of arrays, each (21, 3) with raw MediaPipe x, y, z.
            Length 0, 1 or 2 (extra hands are ignored).

    Returns
    -------
    np.ndarray of shape (129,), float32
    """
    out = np.zeros(FEATURE_COUNT, dtype=np.float32)
    if not hands:
        return out

    hands = [np.asarray(h, dtype=np.float32).reshape(21, 3) for h in hands[:2]]
    hands.sort(key=lambda h: h[0, 0])           # left-to-right by wrist x

    scale_a = _hand_scale(hands[0])
    for i, h in enumerate(hands):
        s = _hand_scale(h)
        rel = (h - h[0]) / s
        out[i * HAND_FEATS:(i + 1) * HAND_FEATS] = rel.reshape(-1)

    if len(hands) == 2:
        out[126:129] = (hands[1][0] - hands[0][0]) / scale_a
    return out


def mirror_hands(hands: list[np.ndarray]) -> list[np.ndarray]:
    """Horizontally flip hands (used as data augmentation)."""
    flipped = []
    for h in hands:
        h = np.asarray(h, dtype=np.float32).reshape(21, 3).copy()
        h[:, 0] = 1.0 - h[:, 0]
        flipped.append(h)
    return flipped
