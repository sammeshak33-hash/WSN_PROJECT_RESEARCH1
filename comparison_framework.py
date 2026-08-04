import os
import pandas as pd
import matplotlib.pyplot as plt


class ComparisonFramework:

    def __init__(self):

        self.results = []

        os.makedirs("Results", exist_ok=True)

    def add_result(
        self,
        model_name,
        accuracy,
        precision,
        recall,
        f1,
        training_time,
        testing_time
    ):

        self.results.append({

            "Model": model_name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1": f1,

            "Training Time": training_time,

            "Testing Time": testing_time

        })

    def create_table(self):

        df = pd.DataFrame(self.results)

        print("\n====================================")
        print("MODEL COMPARISON TABLE")
        print("====================================")

        print(df)

        df.to_csv(

            "Results/model_comparison.csv",

            index=False

        )

        return df

    def accuracy_graph(self):

        df = pd.DataFrame(self.results)

        plt.figure(figsize=(8,5))

        plt.bar(

            df["Model"],

            df["Accuracy"]

        )

        plt.title("Accuracy Comparison")

        plt.ylabel("Accuracy")

        plt.grid(True)

        plt.savefig(

            "Results/accuracy_comparison_models.png"

        )

        plt.show()

    def f1_graph(self):

        df = pd.DataFrame(self.results)

        plt.figure(figsize=(8,5))

        plt.bar(

            df["Model"],

            df["F1"]

        )

        plt.title("F1 Score Comparison")

        plt.grid(True)

        plt.savefig(

            "Results/f1_comparison.png"

        )

        plt.show()

    def training_graph(self):

        df = pd.DataFrame(self.results)

        plt.figure(figsize=(8,5))

        plt.bar(

            df["Model"],

            df["Training Time"]

        )

        plt.title("Training Time Comparison")

        plt.grid(True)

        plt.savefig(

            "Results/training_time.png"

        )

        plt.show()

    def testing_graph(self):

        df = pd.DataFrame(self.results)

        plt.figure(figsize=(8,5))

        plt.bar(

            df["Model"],

            df["Testing Time"]

        )

        plt.title("Testing Time Comparison")

        plt.grid(True)

        plt.savefig(

            "Results/testing_time.png"

        )

        plt.show()