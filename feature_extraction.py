import csv
import random


class FeatureExtraction:

    def __init__(self, nodes):
        self.nodes = nodes
        self.features = []

    def extract_features(self):

        self.features = []

        for node in self.nodes:

            # Packet delivered only if packet was generated
            # and the node is still alive.
            packet_delivered = 1 if (
                node.packet == 1 and node.alive
            ) else 0

            feature = [
                node.id,
                round(node.energy, 4),
                round(node.direct_trust, 4),
                round(node.indirect_trust, 4),
                round(node.total_trust, 4),
                round(node.distance_to_bs, 4),
                node.packet,
                packet_delivered
            ]

            self.features.append(feature)

    def display_features(self):

        print("\n========== Extracted Features ==========\n")

        print("NodeID Energy DTrust ITrust TTrust DistBS Packet Delivered Attack")

        labels = self.inject_attacks()

        for row, label in zip(self.features[:10], labels[:10]):
            print(row + [label])

    def save_dataset(self, filename="network_dataset.csv"):

        header = [
            "NodeID",
            "Energy",
            "DirectTrust",
            "IndirectTrust",
            "TotalTrust",
            "DistanceToBS",
            "Packet",
            "Delivered",
            "AttackLabel"
        ]

        with open(filename, "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow(header)

            labels = self.inject_attacks()

            updated_features = []

            for feature, label in zip(self.features, labels):
                updated_features.append(feature + [label])

            writer.writerows(updated_features)

        print(f"\nDataset saved as {filename}")

    def inject_attacks(self):

        labels = []

        for node in self.nodes:

            r = random.random()

            if r < 0.70:

                label = "Normal"

            elif r < 0.80:

                label = "Blackhole"

                node.total_trust *= 0.30

            elif r < 0.90:

                label = "Grayhole"

                node.total_trust *= 0.55

            else:

                label = "DoS"

                node.energy *= 0.60

            labels.append(label)

        return labels    