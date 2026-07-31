import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    SimpleRNN,
    Concatenate,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.models import Model


class PRCNN:

    def __init__(self, input_shape, num_classes):

        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):

        # ==========================================
        # Input Layer
        # ==========================================

        inputs = Input(shape=self.input_shape)

        # ==========================================
        # CNN Branch
        # ==========================================

        cnn = Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same"
        )(inputs)

        cnn = BatchNormalization()(cnn)

        cnn = MaxPooling1D(pool_size=2)(cnn)

        cnn = Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same"
        )(cnn)

        cnn = BatchNormalization()(cnn)

        cnn = MaxPooling1D(pool_size=2)(cnn)

        cnn = Flatten()(cnn)

        # ==========================================
        # RNN Branch
        # ==========================================

        rnn = SimpleRNN(
            units=64,
            activation="tanh"
        )(inputs)

        # ==========================================
        # Feature Fusion
        # ==========================================

        merged = Concatenate()([
            cnn,
            rnn
        ])

        # ==========================================
        # Fully Connected Layers
        # ==========================================

        dense = Dense(
            128,
            activation="relu"
        )(merged)

        dense = Dropout(0.30)(dense)

        dense = Dense(
            64,
            activation="relu"
        )(dense)

        dense = Dropout(0.20)(dense)

        # ==========================================
        # Output Layer
        # ==========================================

        outputs = Dense(
            self.num_classes,
            activation="softmax"
        )(dense)

        # ==========================================
        # Build Model
        # ==========================================

        model = Model(
            inputs=inputs,
            outputs=outputs,
            name="PRCNN"
        )

        return model