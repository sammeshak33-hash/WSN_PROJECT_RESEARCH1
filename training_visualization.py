import os
import pandas as pd
import matplotlib.pyplot as plt


class TrainingVisualization:

    def __init__(self, filename):

        self.history = pd.read_csv(filename)

        os.makedirs("Results", exist_ok=True)

    # ==========================================
    # Training Accuracy
    # ==========================================

    def training_accuracy(self):

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.history["accuracy"],
            linewidth=2
        )

        plt.title("Training Accuracy")

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            "Results/training_accuracy.png"
        )

        plt.show()

    # ==========================================
    # Validation Accuracy
    # ==========================================

    def validation_accuracy(self):

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.history["val_accuracy"],
            linewidth=2
        )

        plt.title("Validation Accuracy")

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            "Results/validation_accuracy.png"
        )

        plt.show()

    # ==========================================
    # Training Loss
    # ==========================================

    def training_loss(self):

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.history["loss"],
            linewidth=2
        )

        plt.title("Training Loss")

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            "Results/training_loss.png"
        )

        plt.show()

    # ==========================================
    # Validation Loss
    # ==========================================

    def validation_loss(self):

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.history["val_loss"],
            linewidth=2
        )

        plt.title("Validation Loss")

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            "Results/validation_loss.png"
        )

        plt.show()

    # ==========================================
    # Combined Accuracy
    # ==========================================

    def combined_accuracy(self):

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.history["accuracy"],
            label="Training",
            linewidth=2
        )

        plt.plot(
            self.history["val_accuracy"],
            label="Validation",
            linewidth=2
        )

        plt.title("Accuracy Comparison")

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            "Results/accuracy_comparison.png"
        )

        plt.show()

    # ==========================================
    # Combined Loss
    # ==========================================

    def combined_loss(self):

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.history["loss"],
            label="Training",
            linewidth=2
        )

        plt.plot(
            self.history["val_loss"],
            label="Validation",
            linewidth=2
        )

        plt.title("Loss Comparison")

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            "Results/loss_comparison.png"
        )

        plt.show()