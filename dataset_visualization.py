import pandas as pd
import matplotlib.pyplot as plt


class DatasetVisualization:

    def __init__(self, filename):
        self.data = pd.read_csv(filename)

    def attack_distribution(self):

        counts = self.data["AttackLabel"].value_counts()

        plt.figure(figsize=(8, 5))

        counts.plot(kind="bar")

        plt.title("Attack Distribution")
        plt.xlabel("Attack Type")
        plt.ylabel("Number of Samples")

        plt.grid(True)

        plt.savefig("Results/attack_distribution.png")

        plt.show()

    def energy_distribution(self):

        plt.figure(figsize=(8, 5))

        plt.hist(
            self.data["Energy"],
            bins=20
        )

        plt.title("Energy Distribution")
        plt.xlabel("Residual Energy")
        plt.ylabel("Frequency")

        plt.grid(True)

        plt.savefig("Results/energy_distribution.png")

        plt.show()

    def trust_distribution(self):

        plt.figure(figsize=(8, 5))

        plt.hist(
            self.data["TotalTrust"],
            bins=20
        )

        plt.title("Overall Trust Distribution")
        plt.xlabel("Trust")
        plt.ylabel("Frequency")

        plt.grid(True)

        plt.savefig("Results/trust_distribution.png")

        plt.show()

    def correlation_matrix(self):

        correlation = self.data.corr(numeric_only=True)

        plt.figure(figsize=(10, 8))

        plt.imshow(correlation)

        plt.colorbar()

        plt.xticks(
            range(len(correlation.columns)),
            correlation.columns,
            rotation=90
        )

        plt.yticks(
            range(len(correlation.columns)),
            correlation.columns
        )

        plt.title("Feature Correlation Matrix")

        plt.tight_layout()

        plt.savefig("Results/correlation_matrix.png")

        plt.show()