"""
Feature extraction with ResNet50 (pre-trained on ImageNet).

Typical flow:
  extractor = FeatureExtractor()
  features  = extractor.extract_all(image_paths)
  extractor.save("model/features.pkl")
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src.config import IMG_SIZE, MODEL_DIR


class FeatureExtractor:
    """Wrap ResNet50 and expose per-image 2048-d vectors."""

    def __init__(self):
        self.model = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg",
        )

    def extract_one(self, img_path: str) -> np.ndarray:
        """2048-d vector for a single image."""
        img = load_img(img_path, target_size=IMG_SIZE)
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        return self.model.predict(arr, verbose=0).flatten()

    def extract_all(self, img_paths: List[str]) -> Dict[str, np.ndarray]:
        """Map image filename -> 2048 vector for every path."""
        features = {}
        for p in img_paths:
            name = Path(p).name
            features[name] = self.extract_one(p)
        return features

    @staticmethod
    def save(features: Dict[str, np.ndarray],
             path: Optional[Path] = None) -> Path:
        """Persist feature dict to pickle."""
        if path is None:
            path = MODEL_DIR / "features.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(features, f)
        return path

    @staticmethod
    def load(path: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """Load feature dict from pickle."""
        if path is None:
            path = MODEL_DIR / "features.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
