"""
Export the trained Keras model + scaler + label encoder to one NumPy file.

Run from the project root after training:
    python src/export_numpy_model.py
Creates models/isl_numpy.npz, which app/streamlit_app.py and src/predict.py
prefer over the Keras files.
"""
import os
import pickle

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "models")


def main() -> None:
    model = tf.keras.models.load_model(os.path.join(M, "isl_model.h5"))
    with open(os.path.join(M, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(M, "label_encoder.pkl"), "rb") as f:
        enc = pickle.load(f)

    weights = model.get_weights()          # [W0, b0, W1, b1, W2, b2]
    assert len(weights) % 2 == 0
    out = {
        "n_layers": np.array(len(weights) // 2),
        "mean": scaler.mean_.astype(np.float32),
        "scale": scaler.scale_.astype(np.float32),
        "classes": np.array(enc.classes_).astype(str),
    }
    for i in range(len(weights) // 2):
        out[f"W{i}"] = weights[2 * i]
        out[f"b{i}"] = weights[2 * i + 1]
    path = os.path.join(M, "isl_numpy.npz")
    np.savez(path, **out)
    print(f"saved {path}  ({len(weights) // 2} layers, {len(enc.classes_)} classes)")


if __name__ == "__main__":
    main()
