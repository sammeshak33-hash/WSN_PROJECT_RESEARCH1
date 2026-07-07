from deployment import plot_deployment

from energy_model import (
    transmit_packet,
    receive_packet,
    print_energy,
)

from config import PACKET_SIZE

from prcnn_model import PRCNN
from simulation import Simulation

from dataset_generator import DatasetGenerator

from large_dataset_validation import LargeDatasetValidation
from dataset_visualization import DatasetVisualization

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

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
    prep = simulation_data["prep"]


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



   

    # ==========================
    # PRCNN Model
    # ==========================

    input_shape = (prep.X_train.shape[1], 1)

    prcnn = PRCNN(
        input_shape=input_shape,
        num_classes=4
    )

    model = prcnn.build_model()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nPRCNN Model Compiled Successfully")

    model.summary()

    # ==========================
    # PRCNN Model Training
    # ==========================

    print("\n==============================")
    print("PRCNN Model Training Started")
    print("==============================")

    history = model.fit(
        prep.X_train,
        prep.y_train,
        validation_split=0.20,
        epochs=20,
        batch_size=16,
        verbose=1
    )

    print("\n==============================")
    print("Training Completed Successfully")
    print("==============================")

    # ==========================
    # PRCNN Model Testing
    # ==========================

    print("\n==============================")
    print("Testing PRCNN Model")
    print("==============================")

    test_loss, test_accuracy = model.evaluate(
        prep.X_test,
        prep.y_test,
        verbose=1
    )

    print("\nTest Loss :", round(test_loss, 4))
    print("Test Accuracy :", round(test_accuracy * 100, 2), "%")

    # ==========================
    # Predict Attack Classes
    # ==========================

    predictions = model.predict(prep.X_test)

    predicted_classes = predictions.argmax(axis=1)

    # ==========================
    # Performance Evaluation
    # ==========================

    print("\n==============================")
    print("Performance Evaluation")
    print("==============================")

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

    print(f"\nAccuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")

    print("\n==============================")
    print("Sample Predictions")
    print("==============================")

    encoder = prep.encoder

    for i in range(min(10, len(predicted_classes))):

        actual = encoder.inverse_transform(
            [prep.y_test[i]]
        )[0]

        predicted = encoder.inverse_transform(
            [predicted_classes[i]]
        )[0]

        print(f"Sample {i+1}")
        print("Actual    :", actual)
        print("Predicted :", predicted)
        print("-------------------------")

    # ==========================
    # Confusion Matrix
    # ==========================

    cm = confusion_matrix(
        prep.y_test,
        predicted_classes
    )

    print("\n==============================")
    print("Confusion Matrix")
    print("==============================")
    print(cm)

    # ==========================
    # Classification Report
    # ==========================

    print("\n==============================")
    print("Classification Report")
    print("==============================")

    print(
        classification_report(
            prep.y_test,
            predicted_classes,
            target_names=prep.encoder.classes_
        )
    )


if __name__ == "__main__":
    main()