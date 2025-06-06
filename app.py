import argparse
from pathlib import Path
import torch
from torchvision import transforms, models
from PIL import Image

# Load class names manually (ensure it matches your training set)
class_names = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato_Early_blight",
    "Tomato_healthy"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model(model_path="model/plant_cnn.pt"):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

# Predict
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# CLI setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease Classifier")
    parser.add_argument("image_path", type=str, help="Path to input image")
    args = parser.parse_args()

    model = load_model()
    image_tensor = preprocess_image(args.image_path)
    prediction = predict(model, image_tensor)

    print(f"Predicted Disease: {prediction}")
