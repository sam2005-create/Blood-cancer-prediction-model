"""Tkinter desktop app: pick a blood smear image and see the predicted class.

Usage: python gui.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from predict import predict_image


class BloodCancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Blood Cancer Detection")
        self.root.geometry("650x720")
        self.root.minsize(550, 620)

        self.image_path = None
        self.preview = None

        tk.Label(root, text="Blood Cancer Detection", font=("Arial", 20, "bold")).pack(pady=18)

        self.image_label = tk.Label(root, text="No image selected", width=45, height=18, relief="groove")
        self.image_label.pack(pady=12)

        tk.Button(root, text="Upload image", command=self.upload_image, font=("Arial", 12)).pack(pady=8)
        tk.Button(root, text="Predict", command=self.run_prediction, font=("Arial", 12, "bold")).pack(pady=8)

        self.result_label = tk.Label(root, text="Prediction: --", font=("Arial", 15, "bold"))
        self.result_label.pack(pady=(18, 6))

        self.scores_label = tk.Label(root, text="", font=("Arial", 11), justify="left")
        self.scores_label.pack(pady=8)

        tk.Label(
            root, text="Student project. Not for medical diagnosis.", font=("Arial", 9), fg="gray"
        ).pack(side="bottom", pady=12)

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select a blood smear image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((420, 360))
        except Exception as exc:
            messagebox.showerror("Image error", str(exc))
            return

        self.image_path = path
        self.preview = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.preview, text="")
        self.result_label.config(text="Prediction: --")
        self.scores_label.config(text="")

    def run_prediction(self):
        if not self.image_path:
            messagebox.showwarning("No image", "Upload an image first.")
            return
        try:
            label, confidence, scores = predict_image(self.image_path)
        except Exception as exc:
            messagebox.showerror("Prediction error", str(exc))
            return

        self.result_label.config(text=f"Prediction: {label} ({confidence:.1f}%)")
        self.scores_label.config(text="\n".join(f"{name}: {s:.1f}%" for name, s in scores.items()))


if __name__ == "__main__":
    root = tk.Tk()
    BloodCancerApp(root)
    root.mainloop()
