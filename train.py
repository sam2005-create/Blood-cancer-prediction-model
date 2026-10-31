"""Train the MobileNetV2 classifier on the images in dataset/.

Usage: python train.py
"""

import csv
import json

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config


def load_data(dataset_dir, img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE):
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=config.VALIDATION_SPLIT)
    common = dict(
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
    )
    train_data = datagen.flow_from_directory(
        str(dataset_dir), subset="training", shuffle=True, seed=config.SEED, **common
    )
    val_data = datagen.flow_from_directory(
        str(dataset_dir), subset="validation", shuffle=False, **common
    )
    return train_data, val_data


def build_model(num_classes, img_size=config.IMG_SIZE, weights="imagenet"):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights=weights
    )
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base.input, outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def save_history(history, path):
    h = history.history
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "accuracy", "val_accuracy", "loss", "val_loss"])
        for i in range(len(h["accuracy"])):
            writer.writerow([i + 1, h["accuracy"][i], h["val_accuracy"][i], h["loss"][i], h["val_loss"][i]])


def main(dataset_dir=config.DATASET_DIR, model_dir=config.MODEL_DIR, epochs=config.EPOCHS,
         img_size=config.IMG_SIZE, weights="imagenet"):
    if not dataset_dir.is_dir():
        raise SystemExit(f"No dataset folder at {dataset_dir}. See the README for the layout.")

    tf.random.set_seed(config.SEED)
    train_data, val_data = load_data(dataset_dir, img_size)
    print("Classes:", train_data.class_indices)
    print(f"Training images: {train_data.samples}, validation images: {val_data.samples}")

    model = build_model(train_data.num_classes, img_size, weights)
    history = model.fit(train_data, validation_data=val_data, epochs=epochs)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / config.MODEL_PATH.name)

    # Save the label order with the model so predict.py never has to guess it.
    class_names = sorted(train_data.class_indices, key=train_data.class_indices.get)
    (model_dir / config.CLASS_NAMES_PATH.name).write_text(json.dumps(class_names), encoding="utf-8")

    save_history(history, model_dir / config.HISTORY_PATH.name)
    print(f"Saved model, class names and training history to {model_dir}")


if __name__ == "__main__":
    main()
