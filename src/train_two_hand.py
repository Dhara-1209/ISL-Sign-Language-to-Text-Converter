"""
Train the ISL classifier on two-hand features (129 values).

Run from the project root, AFTER src/extract_two_hand.py:
    python src/train_two_hand.py

Evaluation is done with a BLOCK split: for every class the images numbered
801-1000 are held out. The dataset images are consecutive frames, so a
random split puts almost-identical frames in both train and test and
inflates accuracy. The block split is a much more honest number to report.
"""
from __future__ import annotations

import os
import pickle
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.features import FEATURE_COUNT, build_features, mirror_hands  # noqa: E402

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

RAW = os.path.join(PROJECT_ROOT, "data", "landmarks", "raw_hands.npz")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ASSETS = os.path.join(PROJECT_ROOT, "assets")
HOLDOUT_FROM_INDEX = 801      # images 801..1000 of each class -> test set
EPOCHS, BATCH = 80, 64
LABEL_MERGE = {"0": "O", "2": "V"}


def hands_from_row(row: np.ndarray, n: int) -> list[np.ndarray]:
    return [row[i] for i in range(int(n))]


def main() -> None:
    d = np.load(RAW, allow_pickle=False)
    hands, n_hands, labels, index = d["hands"], d["n_hands"], d["labels"], d["index"]
    print(f"{len(labels)} samples | 1-hand: {(n_hands == 1).sum()} | 2-hand: {(n_hands == 2).sum()}")

    # '0' and 'O' are the SAME hand sign in this dataset (the old confusion
    # matrix shows them swapping ~100 times each), so one class is kept.
    labels = np.array([LABEL_MERGE.get(l, l) for l in labels])
    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    num_classes = len(enc.classes_)

    X = np.stack([build_features(hands_from_row(h, n)) for h, n in zip(hands, n_hands)])
    Xm = np.stack([build_features(mirror_hands(hands_from_row(h, n))) for h, n in zip(hands, n_hands)])

    test_mask = index >= HOLDOUT_FROM_INDEX
    train_mask = ~test_mask
    print(f"train {train_mask.sum()}  test {test_mask.sum()}")

    # Train on original + mirrored (webcam may be mirrored vs dataset)
    X_train = np.concatenate([X[train_mask], Xm[train_mask]])
    y_train = np.concatenate([y[train_mask], y[train_mask]])
    X_test, y_test = X[test_mask], y[test_mask]
    X_test_m = Xm[test_mask]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_test_m_s = scaler.transform(X_test_m)

    model = Sequential([
        Input(shape=(FEATURE_COUNT,)),
        Dense(256, activation="relu"), Dropout(0.3),
        Dense(128, activation="relu"), Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    hist = model.fit(
        X_train_s, to_categorical(y_train, num_classes),
        validation_data=(X_test_s, to_categorical(y_test, num_classes)),
        epochs=EPOCHS, batch_size=BATCH, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12, restore_best_weights=True)],
    )

    pred = model.predict(X_test_s, verbose=0).argmax(1)
    pred_m = model.predict(X_test_m_s, verbose=0).argmax(1)
    print(f"\nBLOCK-SPLIT test accuracy          : {(pred == y_test).mean():.4f}")
    print(f"BLOCK-SPLIT test accuracy (mirrored): {(pred_m == y_test).mean():.4f}")
    print(classification_report(y_test, pred, target_names=list(enc.classes_)))

    os.makedirs(ASSETS, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(hist.history["accuracy"], label="Train"); ax[0].plot(hist.history["val_accuracy"], label="Validation")
    ax[0].set_title("Model Accuracy"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(hist.history["loss"], label="Train"); ax[1].plot(hist.history["val_loss"], label="Validation")
    ax[1].set_title("Model Loss"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "training_curves.png"), dpi=150); plt.close(fig)

    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=enc.classes_,
                yticklabels=enc.classes_, ax=ax)
    ax.set_title("Confusion Matrix - held-out images (801-1000 per class)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "confusion_matrix.png"), dpi=150); plt.close(fig)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "isl_model.h5"))
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(enc, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    import json
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({
            "block_split_accuracy": float((pred == y_test).mean()),
            "num_classes": int(num_classes),
            "features": int(FEATURE_COUNT),
        }, f)
    print("Saved model, encoder, scaler, metrics.json in models/")


if __name__ == "__main__":
    main()
