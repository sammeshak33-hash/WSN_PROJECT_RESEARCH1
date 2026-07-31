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

                round(node.energy,4),

                round(node.packet_delivery_ratio,4),

                node.transmission_count,

                node.packet_dropped,

                node.packet_delivered,

                round(node.direct_trust,4),

                round(node.indirect_trust,4),

                round(node.total_trust,4),

                node.attack_type

            ]

            self.features.append(feature)

    def display_features(self):

        print("\n========== Extracted Features ==========\n")

        print(
            "NodeID Energy PDR TxCount Dropped Forwarded "
            "DTrust ITrust TTrust Attack"
        )

        for row in self.features[:10]:

            print(row)

    def save_dataset(self, filename="network_dataset.csv"):

        header = [

            "NodeID",

            "Energy",

            "PacketDeliveryRatio",

            "TransmissionCount",

            "DroppedPackets",

            "ForwardedPackets",

            "DirectTrust",

            "IndirectTrust",

            "TotalTrust",

            "AttackLabel"

        ]

        with open(filename,"w",newline="") as file:

            writer = csv.writer(file)

            writer.writerow(header)

            writer.writerows(self.features)

        print(f"\nDataset saved as {filename}")