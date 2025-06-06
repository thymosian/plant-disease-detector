import torch
from torchvision import models

# Replace with your actual class list
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato_Early_blight",
    "Tomato_healthy"
]


def load_model(model_path):
    model = models.resnet18(weights=None)  # Or newer weight system
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_disease(model, image_tensor, threshold=0.75):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][top_class].item()

        if confidence < threshold:
            return "Unknown or Uncertain", confidence * 100
        return CLASS_NAMES[top_class], confidence * 100

