import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


class EvaluationFramework:

    def __init__(self):
        pass

    def evaluate(
        self,
        model,
        X_test,
        y_test
    ):

        os.makedirs("Results", exist_ok=True)

        # ==========================
        # Testing Time
        # ==========================

        start = time.time()

        predictions = model.predict(X_test)

        end = time.time()

        testing_time = end - start

        # ==========================
        # Predicted Classes
        # ==========================

        predicted_classes = np.argmax(
            predictions,
            axis=1
        )

        # ==========================
        # Metrics
        # ==========================

        accuracy = accuracy_score(
            y_test,
            predicted_classes
        )

        precision = precision_score(
            y_test,
            predicted_classes,
            average="weighted"
        )

        recall = recall_score(
            y_test,
            predicted_classes,
            average="weighted"
        )

        f1 = f1_score(
            y_test,
            predicted_classes,
            average="weighted"
        )

        # ==========================
        # Confusion Matrix
        # ==========================

        cm = confusion_matrix(
            y_test,
            predicted_classes
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot(cmap="Blues")

        plt.title("Confusion Matrix")

        plt.savefig(
            "Results/confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        # ==========================
        # Classification Report
        # ==========================

        print("\n==============================")
        print("CLASSIFICATION REPORT")
        print("==============================")

        print(
            classification_report(
                y_test,
                predicted_classes
            )
        )

        # ==========================
        # Results Dictionary
        # ==========================

        results = {

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1,

            "Testing Time (s)": round(
                testing_time,
                4
            )

        }

        print("\n==============================")
        print("EVALUATION RESULTS")
        print("==============================")

        for key, value in results.items():

            if isinstance(value, float):

                if "Time" in key:

                    print(f"{key}: {value}")

                else:

                    print(f"{key}: {value*100:.2f}%")

            else:

                print(f"{key}: {value}")

        return results