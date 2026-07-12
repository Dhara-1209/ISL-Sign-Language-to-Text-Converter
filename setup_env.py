import cv2
import mediapipe as mp
import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
import matplotlib
import seaborn
import streamlit
import pyttsx3

print("=" * 40)
print("Environment Setup Successful!")
print("=" * 40)

print(f"OpenCV       : {cv2.__version__}")
print(f"MediaPipe    : {mp.__version__}")
print(f"TensorFlow   : {tf.__version__}")
print(f"Scikit-learn : {sklearn.__version__}")
print(f"Pandas       : {pd.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"Matplotlib   : {matplotlib.__version__}")
print(f"Seaborn      : {seaborn.__version__}")
print(f"Streamlit    : {streamlit.__version__}")

print("\nAll libraries imported successfully.")