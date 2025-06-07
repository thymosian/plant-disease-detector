# 🌿 Plant Disease Detector

A deep learning-based tool for detecting plant diseases from leaf images. Upload a photo of a plant leaf and get an instant diagnosis using a fine-tuned ResNet18 model.

🌐 **Live App**: [https://thymosian-plant-disease-detector-streamlit-app-4xnfqg.streamlit.app/](https://thymosian-plant-disease-detector-streamlit-app-4xnfqg.streamlit.app/)

---

## 🎬 Demo
![App Demo](https://raw.githubusercontent.com/thymosian/plant-disease-detector/main/assets/demo.gif)


> A short screen recording showing how to upload a leaf image and get results instantly.

---

## 🚀 Features

- **Model:** Fine-tuned ResNet18, trained on 15 plant disease classes.
- **Web App:** Streamlit interface for easy image upload and diagnosis.
- **CLI:** Command-line interface for batch or scripted predictions.
- **Unknown Detection:** Returns "Unknown or Uncertain" if the model is not confident.
- **Easy Deployment:** Run locally with minimal setup.

---

## 🖼️ Example Classes

- Potato___Early_blight
- Potato___healthy
- Potato___Late_blight
- Tomato_Early_blight
- Tomato_healthy
- *(and more, see [`CLASS_NAMES`](model/utils.py))*

---

## 📦 Project Structure

```
.
├── app.py                  # CLI interface
├── streamlit_app.py        # Streamlit web app
├── model/
│   ├── plant_cnn.pt        # Trained model weights
│   └── utils.py            # Model loading & prediction helpers
├── data/                   # Processed, raw, and subset image data
├── notebooks/
│   └── 01_train_model.ipynb # Model training notebook
├── assets/                 # Images for documentation
├── setup_data.py           # Data preparation script
├── README.md
└── NOTES.md                # Development notes and TODOs
```

---

## ⚡ Quickstart

### 1. Install Requirements

```sh
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```sh
streamlit run streamlit_app.py
```

### 3. Use the CLI

```sh
python app.py path/to/your/image.jpg
```

---

## 🧠 How It Works

- The model is loaded from [`model/plant_cnn.pt`](model/plant_cnn.pt) using [`load_model`](model/utils.py).
- Images are preprocessed to 224×224 pixels.
- [`predict_disease`](model/utils.py) returns the predicted class and confidence.
- If confidence < 75%, the result is "Unknown or Uncertain".

---

## 📝 Training

- See [`notebooks/01_train_model.ipynb`](notebooks/01_train_model.ipynb) for model training details.
- Data is organized in `data/processed/train`, `val`, and `test` folders.

---

## 📌 Future Work

- Add a true "Unknown" class with OOD (out-of-distribution) images.
- Improve generalization with more data augmentation.
- Implement advanced OOD detection methods (ODIN, Mahalanobis, OpenMax).

---

## 📄 License

MIT License

---

_Last updated: 2025-06-06_
