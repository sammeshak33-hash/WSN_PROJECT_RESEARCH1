import pandas as pd


class DatasetValidation:

    def __init__(self, filename):

        self.filename = filename
        self.data = pd.read_csv(filename)

    def dataset_info(self):

        print("\n========== Dataset Information ==========\n")

        print("Total Samples :", len(self.data))
        print()

        print("Total Features :", len(self.data.columns))
        print()

        print("Feature Names")
        print(list(self.data.columns))

    def attack_distribution(self):

        print("\n========== Attack Distribution ==========\n")

        print(self.data["AttackLabel"].value_counts())

    def missing_values(self):

        print("\n========== Missing Values ==========\n")

        print(self.data.isnull().sum())

    def duplicate_rows(self):

        print("\n========== Duplicate Rows ==========\n")

        print("Duplicates :", self.data.duplicated().sum())