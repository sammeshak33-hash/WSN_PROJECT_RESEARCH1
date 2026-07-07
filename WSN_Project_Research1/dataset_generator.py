import pandas as pd


class DatasetGenerator:

    def __init__(self):
        self.final_dataset = pd.DataFrame()

    def append_dataset(self, filename):
        data = pd.read_csv(filename)

        self.final_dataset = pd.concat(
            [self.final_dataset, data],
            ignore_index=True
        )

    def save_dataset(self, filename):
        self.final_dataset.to_csv(
            filename,
            index=False
        )

        print("\n===================================")
        print("Large Dataset Generation Completed")
        print("===================================")
        print("Total Samples :", len(self.final_dataset))