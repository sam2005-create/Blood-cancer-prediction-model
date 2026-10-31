"""Count the images in each class folder before training.

Usage: python check_dataset.py
"""

import config


def count_images(dataset_dir=config.DATASET_DIR):
    if not dataset_dir.is_dir():
        raise SystemExit(f"No dataset folder at {dataset_dir}. See the README for the layout.")

    counts = {}
    for folder in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        counts[folder.name] = sum(
            1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS
        )
    return counts


def main():
    counts = count_images()
    for name in config.CLASS_NAMES:
        print(f"{name}: {counts.get(name, 'missing folder')}")

    # Keras treats every subfolder as a class, so extra folders change the labels.
    extra = sorted(set(counts) - set(config.CLASS_NAMES))
    if extra:
        print("Unexpected folders (remove or rename them):", ", ".join(extra))
    print("Total images:", sum(counts.values()))


if __name__ == "__main__":
    main()
