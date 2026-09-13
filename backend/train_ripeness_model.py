"""
Train a calibrated ripeness model from labeled image folders.

Expected structure:
data/
  ripe/
  unripe/
  rotten/
"""

import os
import cv2
import numpy as np


VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_TO_TARGET_PCT = {
    'unripe': 18.0,
    'ripe': 82.0,
    'rotten': 97.0,
}


def _extract_hs_feature(image):
    image = cv2.resize(image, (256, 256))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = cv2.inRange(sat, 30, 255)
    val_mask = cv2.inRange(val, 35, 245)
    mask = cv2.bitwise_and(mask, val_mask)

    hist = cv2.calcHist([hsv], [0, 1], mask, [24, 24], [0, 180, 0, 256]).flatten().astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist


def _iter_images(folder):
    for name in os.listdir(folder):
        _, ext = os.path.splitext(name.lower())
        if ext in VALID_EXT:
            yield os.path.join(folder, name)


def train_model(data_dir, output_path):
    classes = ['unripe', 'ripe', 'rotten']
    prototypes = []
    class_mean_pct = []
    counts = {}

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        features = []
        for image_path in _iter_images(class_dir):
            image = cv2.imread(image_path)
            if image is None:
                continue
            features.append(_extract_hs_feature(image))

        if not features:
            raise RuntimeError(f"No readable images found in {class_dir}")

        features = np.array(features, dtype=np.float32)
        prototype = np.mean(features, axis=0)
        prototype /= max(np.linalg.norm(prototype), 1e-8)

        prototypes.append(prototype)
        class_mean_pct.append(CLASS_TO_TARGET_PCT[class_name])
        counts[class_name] = len(features)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        classes=np.array(classes),
        prototypes=np.array(prototypes, dtype=np.float32),
        class_mean_pct=np.array(class_mean_pct, dtype=np.float32)
    )
    return counts


if __name__ == "__main__":
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(backend_dir)
    data_dir = os.path.join(root_dir, "data")
    output_path = os.path.join(backend_dir, "models", "ripeness_model.npz")

    counts = train_model(data_dir, output_path)
    print(f"Model saved to: {output_path}")
    print(f"Images used: {counts}")
