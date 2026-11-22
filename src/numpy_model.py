"""
TensorFlow-free inference for the ISL classifier.

The trained Keras MLP is just three Dense layers, so the live app can run it
with plain NumPy. This avoids Keras/TensorFlow threading problems inside
streamlit-webrtc and makes every frame much faster.

models/isl_numpy.npz is created by src/export_numpy_model.py from the
trained Keras model, scaler and label encoder.
"""
from __future__ import annotations

import numpy as np


class NpScaler:
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.scale = scale.astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.scale


class NpEncoder:
    def __init__(self, classes: np.ndarray):
        self.classes_ = np.asarray(classes)

    def inverse_transform(self, idx) -> np.ndarray:
        return self.classes_[np.asarray(idx, dtype=int)]


class NpModel:
    """relu(x W0 + b0) -> relu(. W1 + b1) -> softmax(. W2 + b2)"""

    def __init__(self, weights: list[np.ndarray]):
        self.W = [w.astype(np.float32) for w in weights[0::2]]
        self.b = [b.astype(np.float32) for b in weights[1::2]]

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.maximum(h @ W + b, 0.0)
        z = h @ self.W[-1] + self.b[-1]
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def load_numpy_artifacts(path: str):
    """Return (model, encoder, scaler) with the same interface the app uses."""
    d = np.load(path, allow_pickle=False)
    n_layers = int(d["n_layers"])
    weights = []
    for i in range(n_layers):
        weights += [d[f"W{i}"], d[f"b{i}"]]
    return (NpModel(weights),
            NpEncoder(d["classes"]),
            NpScaler(d["mean"], d["scale"]))
