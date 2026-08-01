from deployment import plot_deployment

from energy_model import (
    transmit_packet,
    receive_packet,
    print_energy,
)

from config import PACKET_SIZE

from simulation import Simulation

from dataset_generator import DatasetGenerator

from large_dataset_validation import LargeDatasetValidation

from dataset_visualization import DatasetVisualization

from preprocessing import Preprocessing

from hyperparameter_tuning import HyperparameterTuner

from training_visualization import TrainingVisualization

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import os
import pandas as pd
import tensorflow as tf

def main():

    generator = DatasetGenerator()

    NUMBER_OF_SIMULATIONS = 10

    simulation_data = None

    for simulation_number in range(NUMBER_OF_SIMULATIONS):

        print(f"\nRunning Simulation {simulation_number + 1}")

        simulation = Simulation()

        simulation_data = simulation.run()

        generator.append_dataset("network_dataset.csv")

    generator.save_dataset("large_network_dataset.csv")

    validator = LargeDatasetValidation("large_network_dataset.csv")

    validator.dataset_info()
    validator.missing_values()
    validator.duplicate_rows()
    validator.attack_distribution()
    validator.statistics()

    visual = DatasetVisualization(
        "large_network_dataset.csv"
    )

    visual.attack_distribution()
    visual.energy_distribution()
    visual.trust_distribution()
    visual.correlation_matrix()

    nodes = simulation_data["nodes"]

    prep = Preprocessing("large_network_dataset.csv")

    prep.separate_features()
    prep.encode_labels()
    prep.normalize()
    prep.split_dataset()

    prep.X_train = prep.X_train.reshape(
        prep.X_train.shape[0],
        prep.X_train.shape[1],
        1
    )

    prep.X_test = prep.X_test.reshape(
        prep.X_test.shape[0],
        prep.X_test.shape[1],
        1
    )

    # ==========================================
    # Create Output Directories
    # ==========================================

    os.makedirs("Models", exist_ok=True)
    os.makedirs("Results", exist_ok=True)


    # ----------------------------
    # Plot Deployment
    # ----------------------------
    plot_deployment(nodes)

    # ----------------------------
    # Display first 10 nodes
    # ----------------------------
    print("\nNode Information\n")

    for node in nodes[:10]:
        print(f"Node {node.id}")
        print(f"Location : ({node.x:.2f}, {node.y:.2f})")
        print(f"Distance to BS : {node.distance_to_bs:.2f}")
        print(f"Neighbors : {node.neighbors}")
        print("-" * 40)

    # ----------------------------
    # Energy Model Test
    # ----------------------------
    print("\n==============================")
    print("ENERGY MODEL TEST")
    print("==============================")

    sender = nodes[0]
    receiver = nodes[1]

    distance = 20  # meters

    print("\nBefore Transmission")
    print_energy(sender)
    print_energy(receiver)

    tx_energy = transmit_packet(sender, PACKET_SIZE, distance)
    rx_energy = receive_packet(receiver, PACKET_SIZE)

    print("\nTransmission Completed")
    print(f"Transmission Energy : {tx_energy:.12f} J")
    print(f"Reception Energy    : {rx_energy:.12f} J")

    print("\nAfter Transmission")
    print_energy(sender)
    print_energy(receiver)



    print("\n====================================")
    print("HYPERPARAMETER OPTIMIZATION")
    print("====================================")

    tuner = HyperparameterTuner()

    best_configuration = tuner.run(prep)

    print("\n====================================")
    print("BEST CONFIGURATION")
    print("====================================")

    print(best_configuration)

    print("\n====================================")
    print("GENERATING TRAINING GRAPHS")
    print("====================================")

    visual = TrainingVisualization(
        "Results/training_history.csv"
    )

    visual.training_accuracy()

    visual.validation_accuracy()

    visual.training_loss()

    visual.validation_loss()

    visual.combined_accuracy()

    visual.combined_loss()

    print("\nAll Training Graphs Generated Successfully")

    # ==========================================
    # Load Best Model
    # ==========================================

    print("\n====================================")
    print("MODEL EVALUATION")
    print("====================================")

    model = tf.keras.models.load_model(
        "Models/prcnn_large_dataset.keras"
    )

    # Predict

    predictions = model.predict(prep.X_test)

    predicted_classes = predictions.argmax(axis=1)

    # ==========================================
    # Performance Metrics
    # ==========================================

    accuracy = accuracy_score(
        prep.y_test,
        predicted_classes
    )

    precision = precision_score(
        prep.y_test,
        predicted_classes,
        average="weighted"
    )

    recall = recall_score(
        prep.y_test,
        predicted_classes,
        average="weighted"
    )

    f1 = f1_score(
        prep.y_test,
        predicted_classes,
        average="weighted"
    )

    print(f"\nAccuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")

    # ==========================================
    # Confusion Matrix
    # ==========================================

    cm = confusion_matrix(
        prep.y_test,
        predicted_classes
    )

    print("\n====================================")
    print("CONFUSION MATRIX")
    print("====================================")

    print(cm)

    # ==========================================
    # Classification Report
    # ==========================================

    print("\n====================================")
    print("CLASSIFICATION REPORT")
    print("====================================")

    print(
        classification_report(
            prep.y_test,
            predicted_classes,
            target_names=prep.encoder.classes_
        )
    )

   


if __name__ == "__main__":
    main()