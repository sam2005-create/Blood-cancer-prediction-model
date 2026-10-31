"""Plot accuracy and loss per epoch from model/training_history.csv.

Usage: python plot_graph.py
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def plot(epochs, train, val, metric, out_path):
    plt.figure()
    plt.plot(epochs, train, marker="o", label=f"Training {metric}")
    plt.plot(epochs, val, marker="o", label=f"Validation {metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"Training and validation {metric}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(history_path=config.HISTORY_PATH, results_dir=config.RESULTS_DIR):
    if not history_path.exists():
        raise SystemExit(f"{history_path} not found. Run train.py first.")

    with open(history_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("The training history file is empty.")

    epochs = [int(r["epoch"]) for r in rows]
    results_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("accuracy", "loss"):
        train = [float(r[metric]) for r in rows]
        val = [float(r[f"val_{metric}"]) for r in rows]
        plot(epochs, train, val, metric, results_dir / f"{metric}.png")

    print(f"Saved accuracy.png and loss.png to {results_dir}")


if __name__ == "__main__":
    main()
