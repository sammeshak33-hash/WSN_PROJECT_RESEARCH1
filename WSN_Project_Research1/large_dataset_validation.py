import pandas as pd


class LargeDatasetValidation:

    def __init__(self, filename):
        self.filename = filename
        self.data = pd.read_csv(filename)

    def dataset_info(self):
        print("\n==============================")
        print("LARGE DATASET INFORMATION")
        print("==============================")
        print("Total Samples :", len(self.data))
        print("Total Features :", len(self.data.columns))

        print("\nColumn Names:")
        for column in self.data.columns:
            print("-", column)

    def missing_values(self):
        print("\n==============================")
        print("MISSING VALUES")
        print("==============================")
        print(self.data.isnull().sum())

    def duplicate_rows(self):
        duplicates = self.data.duplicated().sum()

        print("\n==============================")
        print("DUPLICATE RECORDS")
        print("==============================")
        print("Duplicate Rows :", duplicates)

    def attack_distribution(self):
        print("\n==============================")
        print("ATTACK DISTRIBUTION")
        print("==============================")
        print(self.data["AttackLabel"].value_counts())

    def statistics(self):
        print("\n==============================")
        print("FEATURE STATISTICS")
        print("==============================")
        print(self.data.describe())