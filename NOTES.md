# 📝 Plant Disease Classifier – Notes

## ✅ Current Features

- **Model**: `ResNet18` fine-tuned on 15 plant disease classes.
- **Frontend**: Streamlit interface (`streamlit_app.py`) for uploading and analyzing plant leaf images.
- **Backend**: 
  - `utils.py` handles model loading and prediction.
  - `app.py` provides CLI functionality.
- **Deployment**: Local Streamlit web app.

---

## 🧠 Latest Fix: Handling Unknown Classes

### Problem
The model made confident predictions even when shown images of plants or objects it wasn’t trained on.

### Solution
- Added a **confidence threshold** to `predict_disease()` in `utils.py`.
- If the model's top softmax probability is below the threshold (default: `0.75`), it returns `"Unknown or Uncertain"` instead of a known class.
- Streamlit UI updated to reflect this: when the model is unsure, it notifies the user clearly.

### Code Changes
- `utils.py`: Modified `predict_disease(model, image_tensor, threshold=0.75)` to include threshold logic.
- `streamlit_app.py`: Displays a message if confidence is below threshold.

---

## 📌 Future Improvements

### 1. Add an "Unknown" Class
- Collect diverse images outside training distribution:
  - Other plant species
  - Backgrounds (soil, hands, sky)
  - Random noise / unrelated items
- Train the model with these under a new `"Unknown"` label.

### 2. Improve Generalization
- Apply aggressive data augmentation:
  - Vary lighting, orientation, scale
  - Add noise, blur, background clutter

### 3. Implement True OOD Detection
Explore better out-of-distribution detection methods:
- **ODIN** (temperature scaling + input perturbation)
- **Mahalanobis distance-based detection**
- **OpenMax**
- **Contrastive learning + KNN**

### 4. Model Logging & Feedback
- Log prediction results and confidence scores for future analysis.
- Identify recurring failure patterns to refine threshold or retrain.

---

## 🛠️ Miscellaneous

- **Model path**: `model/plant_cnn.pt`
- **Class definitions**: `CLASS_NAMES` list in `utils.py`
- **Default input image size**: 224×224
- **Device**: GPU (`cuda`) if available; falls back to CPU

---

_Last updated: 2025-06-06_
