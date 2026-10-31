# Blood cancer prediction model

A MobileNetV2 image classifier, built with TensorFlow/Keras, that sorts microscope blood smear images into four classes: Benign, Early, Pre and Pro. It comes with a command-line predictor and a small Tkinter desktop app.

This is a student project, not a medical tool. Don't use its output for diagnosis.

## How it works

- The images are resized to 224x224 and scaled to 0-1. 20% of each class is held out for validation.
- MobileNetV2, pretrained on ImageNet, is used as a frozen feature extractor. On top of it sit global average pooling, a 128-unit ReLU layer and a 4-way softmax.
- Training uses Adam and categorical cross-entropy for 5 epochs with batch size 32.
- `train.py` saves the label order to `model/class_names.json` next to the model, so `predict.py` never mixes up classes.
- `predict.py` reads the image with OpenCV and resizes it the same way Keras does during training (nearest-neighbour), so a prediction sees the same kind of input the model was trained on.

## Files

| File | What it does |
|---|---|
| `config.py` | Paths, image size, epochs and class names in one place |
| `check_dataset.py` | Counts the images per class and warns about stray folders |
| `train.py` | Trains the model and saves it with its training history |
| `evaluate.py` | Accuracy, per-class precision/recall and a confusion matrix on the validation split |
| `plot_graph.py` | Accuracy and loss curves per epoch |
| `predict.py` | Classifies one image from the command line |
| `gui.py` | Desktop app: pick an image, see the prediction and the score for each class |
| `tests/` | An end-to-end test on a tiny fake dataset |

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Dataset

The images aren't in this repo because they're too large. Put them in `dataset/` with one folder per class:

```
dataset/
    Benign/
    Early/
    Pre/
    Pro/
```

Keras takes the class names from these folder names, so keep them exactly like this. Then check the counts:

```bash
python check_dataset.py
```

## Running it

```bash
python train.py              # trains and saves model/blood_cancer_model.keras
python evaluate.py           # writes results/confusion_matrix.png and classification_report.txt
python plot_graph.py         # writes results/accuracy.png and results/loss.png
python predict.py path/to/image.jpg
python gui.py
```

The trained weights aren't committed either, so run `train.py` before predicting.

## Tests

```bash
python -m pytest
```

The test builds a small set of random images, trains for one epoch with an untrained network, and then runs prediction, evaluation and plotting on the result. It checks that the scripts fit together, not how accurate the model is. GitHub Actions runs it on every push.

## Limitations and next steps

- There's no separate test set; the validation split is the only held-out data. A third split would give a fairer number.
- Images are scaled to 0-1, while MobileNetV2 was pretrained on images scaled to -1 to 1. Switching to `mobilenet_v2.preprocess_input` would probably help.
- MobileNetV2 stays frozen the whole time. Unfreezing the top layers for a few more epochs at a low learning rate is the usual next step.
- No data augmentation yet (flips, rotations).
