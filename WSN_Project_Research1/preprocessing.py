import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Preprocessing:

    def __init__(self, filename):
        self.data = pd.read_csv(filename)

    def separate_features(self):

        self.X = self.data.drop(
            columns=["AttackLabel"]
        )

        self.y = self.data["AttackLabel"]

        print("\nFeatures Shape :", self.X.shape)
        print("Labels Shape :", self.y.shape)

    def encode_labels(self):

        encoder = LabelEncoder()

        self.y = encoder.fit_transform(self.y)

        self.encoder = encoder

        print("\nAttack Classes")
        print(encoder.classes_)

    def normalize(self):

        scaler = MinMaxScaler()

        self.X = scaler.fit_transform(self.X)

        self.scaler = scaler

        print("\nNormalization Completed")

    def split_dataset(self):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.20,
            random_state=42,
            stratify=self.y
        )

        print("\nTraining Samples :", len(self.X_train))
        print("Testing Samples :", len(self.X_test))