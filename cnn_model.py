from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)


class CNNModel:

    def __init__(self, input_shape, num_classes):

        self.input_shape = input_shape

        self.num_classes = num_classes

    def build_model(self):

        model = Sequential()

        # ==========================================
        # Input Layer
        # ==========================================

        model.add(

            Input(
                shape=self.input_shape
            )

        )

        # ==========================================
        # First Convolution
        # ==========================================

        model.add(

            Conv1D(

                filters=32,

                kernel_size=3,

                activation="relu",

                padding="same"

            )

        )

        # ==========================================
        # First Max Pooling
        # ==========================================

        model.add(

            MaxPooling1D(

                pool_size=2

            )

        )

        # ==========================================
        # Second Convolution
        # ==========================================

        model.add(

            Conv1D(

                filters=64,

                kernel_size=3,

                activation="relu",

                padding="same"

            )

        )

        # ==========================================
        # Flatten
        # ==========================================

        model.add(

            Flatten()

        )

        # ==========================================
        # Dense Layer
        # ==========================================

        model.add(

            Dense(

                128,

                activation="relu"

            )

        )

        # ==========================================
        # Dropout
        # ==========================================

        model.add(

            Dropout(

                0.5

            )

        )

        # ==========================================
        # Output Layer
        # ==========================================

        model.add(

            Dense(

                self.num_classes,

                activation="softmax"

            )

        )

        # ==========================================
        # Compile Model
        # ==========================================

        model.compile(

            optimizer="adam",

            loss="sparse_categorical_crossentropy",

            metrics=["accuracy"]

        )

        return model