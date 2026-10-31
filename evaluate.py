"""Score the trained model on the validation split and save a confusion matrix.

Usage: python evaluate.py

The validation split is the same 20% of images that train.py held out.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

import config
from predict import load_model
from train import load_data


def main(dataset_dir=config.DATASET_DIR, model_path=config.MODEL_PATH, results_dir=config.RESULTS_DIR):
    model, class_names = load_model(model_path)
    _, val_data = load_data(dataset_dir, img_size=model.input_shape[1])

    y_true = val_data.classes
    y_pred = np.argmax(model.predict(val_data, verbose=0), axis=1)

    accuracy = float(np.mean(y_true == y_pred))
    report = classification_report(
        y_true, y_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0
    )
    print(f"Validation accuracy: {accuracy:.2%} on {len(y_true)} images\n")
    print(report)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "classification_report.txt").write_text(
        f"Validation accuracy: {accuracy:.2%} on {len(y_true)} images\n\n{report}", encoding="utf-8"
    )

    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(cmap="Blues", colorbar=False)
    plt.title("Confusion matrix (validation set)")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=200)
    plt.close()
    print(f"Saved the report and confusion matrix to {results_dir}")
    return accuracy


if __name__ == "__main__":
    main()
