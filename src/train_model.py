import os
import sys
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # head-less backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH     = os.path.join(PROJECT_ROOT, "data", "landmarks", "isl_landmarks.csv")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
ASSETS_DIR   = os.path.join(PROJECT_ROOT, "assets")
MODEL_PATH   = os.path.join(MODEL_DIR, "isl_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
CURVES_PATH  = os.path.join(ASSETS_DIR, "training_curves.png")
CM_PATH      = os.path.join(ASSETS_DIR, "confusion_matrix.png")

FEATURE_COUNT = 63
NUM_CLASSES   = 36
EPOCHS        = 60
BATCH_SIZE    = 32


def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load landmark CSV and return features (X) and labels (y).

    Parameters
    ----------
    csv_path : str
        Path to isl_landmarks.csv.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X with shape (n_samples, 63) and y with shape (n_samples,).
    """
    print(f"[1/8] Loading data from {csv_path} …")
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :FEATURE_COUNT].values.astype(np.float32)
    y = df["label"].values
    print(f"      → {len(df)} rows, {FEATURE_COUNT} features, "
          f"{len(np.unique(y))} classes")
    return X, y


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Fit a LabelEncoder on string labels and one-hot encode for Keras.

    Parameters
    ----------
    y : np.ndarray
        1-D array of string labels.

    Returns
    -------
    tuple
        (one-hot array, integer-encoded array, fitted LabelEncoder)
    """
    print("[2/8] Encoding labels …")
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(y)
    y_oh = to_categorical(y_int, num_classes=NUM_CLASSES)
    print(f"      → Classes: {list(encoder.classes_)}")
    return y_oh, y_int, encoder


def split_and_scale(
    X: np.ndarray, y_oh: np.ndarray, y_int: np.ndarray
) -> tuple[np.ndarray, ...]:
    """
    Stratified 80/20 split then StandardScaler on features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y_oh : np.ndarray
        One-hot encoded labels.
    y_int : np.ndarray
        Integer-encoded labels (for stratify).

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler)
    """
    print("[3/8] Splitting data (80/20, stratified, seed=42) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_oh,
        test_size=0.20,
        random_state=SEED,
        stratify=y_int,
    )
    print(f"      → Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    print("[4/8] Scaling features with StandardScaler …")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def build_model() -> Sequential:
    """
    Build the Keras MLP:
        Input(63) → Dense(128, relu) → Dropout(0.3)
                  → Dense(64, relu)  → Dropout(0.3)
                  → Dense(36, softmax)

    Returns
    -------
    Sequential
        Compiled Keras model.
    """
    print("[5/8] Building Keras model …")
    model = Sequential([
        Input(shape=(FEATURE_COUNT,)),
        Dense(128, activation="relu", name="dense_1"),
        Dropout(0.3, name="dropout_1"),
        Dense(64, activation="relu", name="dense_2"),
        Dropout(0.3, name="dropout_2"),
        Dense(NUM_CLASSES, activation="softmax", name="output"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tf.keras.callbacks.History:
    """
    Train the model for EPOCHS epochs with BATCH_SIZE.

    Parameters
    ----------
    model : Sequential
        Compiled Keras model.
    X_train, y_train : np.ndarray
        Training data.
    X_test, y_test : np.ndarray
        Validation data.

    Returns
    -------
    History
        Keras training history object.
    """
    print(f"[6/8] Training for {EPOCHS} epochs (batch_size={BATCH_SIZE}) …")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1,
    )
    return history


def evaluate_and_plot(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_int_test: np.ndarray,
    encoder: LabelEncoder,
    history: tf.keras.callbacks.History,
) -> None:
    """
    Evaluate model, print classification report, and save
    training curves + confusion matrix plots.

    Parameters
    ----------
    model : Sequential
        Trained Keras model.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        One-hot test labels.
    y_int_test : np.ndarray
        Integer test labels.
    encoder : LabelEncoder
        Fitted encoder for class names.
    history : History
        Training history.
    """
    print("[7/8] Evaluating model & generating plots …")

    # ── Test loss / accuracy ─────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"      → Test accuracy: {test_acc:.4f}")
    print(f"      → Test loss    : {test_loss:.4f}")

    # ── Classification report ────────────────────────────────
    y_pred_oh  = model.predict(X_test, verbose=0)
    y_pred_int = np.argmax(y_pred_oh, axis=1)
    class_names = list(encoder.classes_)

    print("\n      Classification Report:")
    print(classification_report(
        y_int_test, y_pred_int, target_names=class_names
    ))

    # ── Training curves ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=150)
    plt.close(fig)
    print(f"      → Training curves saved to {CURVES_PATH}")

    # ── Confusion matrix ─────────────────────────────────────
    cm = confusion_matrix(y_int_test, y_pred_int)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
    )
    ax.set_title("Confusion Matrix — ISL Gesture Classification", fontsize=16)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=150)
    plt.close(fig)
    print(f"      → Confusion matrix saved to {CM_PATH}")


def save_artifacts(
    model: Sequential,
    encoder: LabelEncoder,
    scaler: StandardScaler,
) -> None:
    """Save model (.h5), label encoder (.pkl), and scaler (.pkl)."""
    print("[8/8] Saving artifacts …")
    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(MODEL_PATH)
    print(f"      → Model saved   : {MODEL_PATH}")

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    print(f"      → Encoder saved : {ENCODER_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"      → Scaler saved  : {SCALER_PATH}")


# ── Main pipeline ────────────────────────────────────────────
def main() -> None:
    """Run the full training pipeline end-to-end."""
    print("\n" + "=" * 60)
    print("  ISL Gesture Classification — Model Training")
    print("=" * 60 + "\n")

    os.makedirs(ASSETS_DIR, exist_ok=True)

    X, y = load_data(CSV_PATH)
    y_oh, y_int, encoder = encode_labels(y)

    # We need integer labels for stratify *before* one-hot
    X_train, X_test, y_train_oh, y_test_oh, scaler = split_and_scale(X, y_oh, y_int)

    # Also keep integer versions for evaluation
    y_train_int = np.argmax(y_train_oh, axis=1)
    y_test_int  = np.argmax(y_test_oh, axis=1)

    model = build_model()
    history = train_model(model, X_train, y_train_oh, X_test, y_test_oh)

    evaluate_and_plot(model, X_test, y_test_oh, y_test_int, encoder, history)
    save_artifacts(model, encoder, scaler)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n✗ Fatal error: {exc}")
        sys.exit(1)