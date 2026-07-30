import csv


class FeatureExtraction:

    def __init__(self, nodes):

        self.nodes = nodes
        self.features = []

    def extract_features(self):

        self.features = []

        for node in self.nodes:

            feature = [
                node.id,
                round(node.energy, 4),
                round(node.direct_trust, 4),
                round(node.indirect_trust, 4),
                round(node.total_trust, 4),
                round(node.distance_to_bs, 4),
                node.packet,
                node.packet_delivered,
                node.attack_type
            ]

            self.features.append(feature)

    def display_features(self):

        print("\n========== Extracted Features ==========\n")

        print(
            "NodeID Energy DTrust ITrust TTrust DistBS Packet Delivered Attack"
        )

        for row in self.features[:10]:
            print(row)

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

            writer.writerows(self.features)

        print(f"\nDataset saved as {filename}")