import os
import time
import pandas as pd
import tensorflow as tf

from prcnn_model import PRCNN


class HyperparameterTuner:

    def __init__(self):

        self.results = []

        self.EPOCHS = [20, 30, 50, 100]

        self.BATCH_SIZES = [16, 32, 64]

        self.LEARNING_RATES = [
            0.001,
            0.0005,
            0.0001
        ]

    def run(self, prep):

        input_shape = (
            prep.X_train.shape[1],
            1
        )

        os.makedirs("Results", exist_ok=True)
        os.makedirs("Models", exist_ok=True)

        for epoch in self.EPOCHS:

            for batch in self.BATCH_SIZES:

                for lr in self.LEARNING_RATES:

                    print("\n====================================")
                    print("Testing Configuration")
                    print("====================================")
                    print(f"Epochs        : {epoch}")
                    print(f"Batch Size    : {batch}")
                    print(f"Learning Rate : {lr}")
                    print("====================================")

                    # -----------------------------
                    # Build Fresh PRCNN
                    # -----------------------------

                    prcnn = PRCNN(
                        input_shape=input_shape,
                        num_classes=4
                    )

                    model = prcnn.build_model()

                    optimizer = tf.keras.optimizers.Adam(
                        learning_rate=lr
                    )

                    model.compile(
                        optimizer=optimizer,
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"]
                    )

                    # -----------------------------
                    # Training
                    # -----------------------------

                    start_time = time.time()

                    history = model.fit(

                        prep.X_train,

                        prep.y_train,

                        validation_data=(

                            prep.X_test,

                            prep.y_test

                        ),

                        epochs=epoch,

                        batch_size=batch,

                        verbose=0

                    )

                    end_time = time.time()

                    training_time = end_time - start_time

                    # -----------------------------
                    # Save Individual Model
                    # -----------------------------

                    model_name = (
                        f"Models/"
                        f"prcnn_E{epoch}"
                        f"_B{batch}"
                        f"_LR{lr}.keras"
                    )

                    model.save(model_name)

                    # -----------------------------
                    # Store Results
                    # -----------------------------

                    self.results.append({

                        "Epochs": epoch,

                        "Batch Size": batch,

                        "Learning Rate": lr,

                        "Training Accuracy":
                            history.history["accuracy"][-1],

                        "Validation Accuracy":
                            history.history["val_accuracy"][-1],

                        "Training Loss":
                            history.history["loss"][-1],

                        "Validation Loss":
                            history.history["val_loss"][-1],

                        "Training Time (s)":
                            round(training_time, 2)

                    })

                    print(
                        "Validation Accuracy :",
                        round(
                            history.history["val_accuracy"][-1] * 100,
                            2
                        ),
                        "%"
                    )

        # =====================================
        # Save CSV
        # =====================================

        df = pd.DataFrame(self.results)

        df.to_csv(

            "Results/hyperparameter_results.csv",

            index=False

        )

        print("\n====================================")
        print("Hyperparameter Results Saved")
        print("Location : Results/hyperparameter_results.csv")
        print("====================================")

        # =====================================
        # Find Best Configuration
        # =====================================

        best = df.loc[
            df["Validation Accuracy"].idxmax()
        ]

        print("\n====================================")
        print("BEST CONFIGURATION")
        print("====================================")

        print(best)

        print("\n====================================")

        return best 