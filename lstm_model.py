from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout
)


class LSTMModel:

    def __init__(
        self,
        input_shape,
        num_classes
    ):

        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):

        model = Sequential()

        model.add(
            Input(shape=self.input_shape)
        )

        model.add(
            LSTM(
                64,
                activation="tanh"
            )
        )

        model.add(
            Dropout(0.5)
        )

        model.add(
            Dense(
                64,
                activation="relu"
            )
        )

        model.add(
            Dropout(0.5)
        )

        model.add(
            Dense(
                self.num_classes,
                activation="softmax"
            )
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model