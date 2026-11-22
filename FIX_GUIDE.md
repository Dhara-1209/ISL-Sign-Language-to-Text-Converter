# ISL project fix – what to do tonight

## 1. Copy these files into your project (same folders, overwrite)
- src/features.py            (new)  – one feature builder shared by training and live app
- src/extract_two_hand.py    (new)  – only needed if you want to re-extract landmarks
- src/train_two_hand.py      (new)  – new training script
- src/predict.py             (changed)
- app/streamlit_app.py       (changed)
- data/landmarks/raw_hands.npz (new) – already extracted, so you can SKIP extraction

## 2. Retrain (about 1-3 minutes on CPU)
    python src/train_two_hand.py

This overwrites models/isl_model.h5, label_encoder.pkl, scaler.pkl,
assets/training_curves.png, assets/confusion_matrix.png and writes
models/metrics.json. Note the "BLOCK-SPLIT test accuracy" it prints.

## 3. Run
    streamlit run app/streamlit_app.py
    (or)  python src/predict.py

## What was actually wrong
1. Many ISL letters (A, D, E, F, H, K, M, N, S, T ...) are TWO-handed. The code used
   num_hands=1, so those signs were reduced to a single arbitrary hand.
2. The live app sharpened the frame (CLAHE) but training images were never sharpened.
3. Hand size was never normalised, so distance from the camera changed the numbers.
4. 0 and O are the same hand sign, so the model can never separate them. Now one class.
5. The reported 93.8% / 98.27% came from a random split of near-duplicate consecutive
   frames. On a fair split (images 801-1000 of each class held out) the old approach
   scored ~77%.

## Honest numbers measured here (scikit-learn MLP, same features)
- Old features, random split (how the report measured): 97.6%
- Old features, fair block split: 76.6%
- New two-hand features, fair block split, 0/O merged: ~86%
Your Keras numbers will be close to these. Quote whatever train_two_hand.py prints.

## Report edits needed
- Replace 93.8% / 87.4% / 98.27% with the number from train_two_hand.py (say "held-out
  images, 801-1000 of each class").
- Replace "63 features" with "129 features (two hands)" and "36 classes" with "35 classes (0 and O merged)".
- Remove or soften "no signs of overfitting" and "24 FPS", "Hugging Face deployment",
  "camera index 1 fallback" unless you actually measured / did them.
- Move "multi-hand detection for two-handed signs" out of future work: it is now done.
