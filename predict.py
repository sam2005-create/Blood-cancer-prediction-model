"""Classify one blood smear image with the trained model.

Usage: python predict.py path/to/image.jpg
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

import config

# Loaded models, keyed by path, so the GUI doesn't reload the model on every click.
_cache = {}


def load_model(model_path=config.MODEL_PATH):
    model_path = Path(model_path).resolve()
    if model_path not in _cache:
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model at {model_path}. Run train.py first.")
        model = tf.keras.models.load_model(str(model_path))

        names_file = model_path.parent / config.CLASS_NAMES_PATH.name
        if names_file.exists():
            class_names = json.loads(names_file.read_text(encoding="utf-8"))
        else:
            class_names = config.CLASS_NAMES
        _cache[model_path] = (model, class_names)
    return _cache[model_path]


def prepare_image(image_path, img_size=config.IMG_SIZE):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Couldn't read {image_path} as an image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Keras resizes training images with nearest-neighbour, so do the same here.
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_NEAREST_EXACT)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def predict_image(image_path, model_path=config.MODEL_PATH):
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, class_names = load_model(model_path)
    img_size = model.input_shape[1]
    probs = model.predict(prepare_image(image_path, img_size), verbose=0)[0]

    scores = {name: float(p) * 100 for name, p in zip(class_names, probs)}
    label = class_names[int(np.argmax(probs))]
    return label, scores[label], scores


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict.py path/to/image.jpg")

    label, confidence, scores = predict_image(sys.argv[1])
    print(f"Prediction: {label} ({confidence:.2f}%)")
    for name, score in scores.items():
        print(f"  {name}: {score:.2f}%")
