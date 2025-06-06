import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw/PlantVillage")
SUBSET_DIR = Path("data/subset")
PROCESSED_DIR = Path("data/processed")
SELECTED_CLASSES = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato_Early_blight",
    "Tomato_healthy"
]

SAMPLES_PER_CLASS = 200  # Adjustable

def create_subset():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"{RAW_DIR} not found.")

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    for cls in SELECTED_CLASSES:
        src = RAW_DIR / cls
        dst = SUBSET_DIR / cls
        dst.mkdir(parents=True, exist_ok=True)

        images = list(src.glob("*.jpg"))
        selected = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))

        for img_path in selected:
            shutil.copy(img_path, dst / img_path.name)

    print(f"[✓] Subset created in {SUBSET_DIR}")

def split_subset(train_ratio=0.7, val_ratio=0.15):
    for split in ["train", "val", "test"]:
        for cls in SELECTED_CLASSES:
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in SELECTED_CLASSES:
        imgs = list((SUBSET_DIR / cls).glob("*.jpg"))
        random.shuffle(imgs)

        n_train = int(len(imgs) * train_ratio)
        n_val = int(len(imgs) * val_ratio)

        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train + n_val],
            "test": imgs[n_train + n_val:],
        }

        for split, files in splits.items():
            for f in files:
                shutil.copy(f, PROCESSED_DIR / split / cls / f.name)

    print(f"[✓] Data split into train/val/test in {PROCESSED_DIR}")

if __name__ == "__main__":
    create_subset()
    split_subset()
