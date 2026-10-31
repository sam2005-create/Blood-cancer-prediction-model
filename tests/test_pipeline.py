"""End-to-end check on a tiny fake dataset: train, save, predict, evaluate.

Uses random images and an untrained MobileNetV2 (weights=None) so it runs
offline in under a minute. It checks the scripts work together, not accuracy.
"""

import json

import numpy as np
import pytest
from PIL import Image

import check_dataset
import config
import evaluate
import plot_graph
import predict
import train

IMG = 96


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    dataset = root / "dataset"
    rng = np.random.default_rng(0)
    for name in config.CLASS_NAMES:
        (dataset / name).mkdir(parents=True)
        for i in range(10):
            pixels = rng.integers(0, 256, (120, 140, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(dataset / name / f"{i}.png")

    model_dir = root / "model"
    train.main(dataset_dir=dataset, model_dir=model_dir, epochs=1, img_size=IMG, weights=None)
    return root, dataset, model_dir


def test_training_saves_model_labels_and_history(trained):
    _, _, model_dir = trained
    assert (model_dir / "blood_cancer_model.keras").exists()
    assert json.loads((model_dir / "class_names.json").read_text()) == config.CLASS_NAMES
    lines = (model_dir / "training_history.csv").read_text().splitlines()
    assert lines[0] == "epoch,accuracy,val_accuracy,loss,val_loss"
    assert len(lines) == 2


def test_predict_returns_a_class_and_scores(trained):
    _, dataset, model_dir = trained
    label, confidence, scores = predict.predict_image(
        dataset / "Pre" / "0.png", model_path=model_dir / "blood_cancer_model.keras"
    )
    assert label in config.CLASS_NAMES
    assert list(scores) == config.CLASS_NAMES
    assert sum(scores.values()) == pytest.approx(100, abs=0.01)
    assert confidence == max(scores.values())


def test_predict_rejects_missing_and_broken_files(trained, tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.predict_image(tmp_path / "nope.png")
    broken = tmp_path / "broken.png"
    broken.write_text("not an image")
    with pytest.raises(ValueError):
        predict.prepare_image(broken)


def test_prediction_input_matches_training_input(trained):
    # predict.py should hand the model exactly what Keras fed it during training.
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    _, dataset, _ = trained
    path = dataset / "Benign" / "3.png"
    keras_img = img_to_array(load_img(path, target_size=(IMG, IMG), interpolation="nearest")) / 255.0
    ours = predict.prepare_image(path, IMG)[0]
    assert ours.shape == keras_img.shape
    assert np.abs(ours - keras_img).max() < 1e-6


def test_evaluate_and_plots_write_results(trained):
    root, dataset, model_dir = trained
    results = root / "results"
    accuracy = evaluate.main(dataset, model_dir / "blood_cancer_model.keras", results)
    assert 0 <= accuracy <= 1
    plot_graph.main(model_dir / "training_history.csv", results)
    for name in ("confusion_matrix.png", "classification_report.txt", "accuracy.png", "loss.png"):
        assert (results / name).exists()


def test_check_dataset_counts_images(trained):
    _, dataset, _ = trained
    assert check_dataset.count_images(dataset) == {name: 10 for name in config.CLASS_NAMES}
