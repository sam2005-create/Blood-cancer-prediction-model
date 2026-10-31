from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"

MODEL_PATH = MODEL_DIR / "blood_cancer_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
HISTORY_PATH = MODEL_DIR / "training_history.csv"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
VALIDATION_SPLIT = 0.2
SEED = 42

# Keras sorts class folders alphabetically, so this is the order the model learns.
CLASS_NAMES = ["Benign", "Early", "Pre", "Pro"]

# Same formats Keras picks up from the class folders.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff"}
