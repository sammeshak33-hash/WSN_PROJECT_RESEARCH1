import random


class PacketTransmission:

    def __init__(self, nodes):

        self.nodes = nodes

        self.total_packets = 0
        self.delivered_packets = 0

    def generate_packets(self):
        """
        Randomly generate packets.
        0 -> No packet
        1 -> Packet generated
        """

        for node in self.nodes:

            node.packet = random.randint(0, 1)

    def transmit_packets(self, trusted_nodes):
        """
        Transmit packets through trusted nodes.
        """

        self.total_packets = 0
        self.delivered_packets = 0

        for node in self.nodes:

            if node.packet == 1:

                self.total_packets += 1

                if node.id in trusted_nodes:

                    self.delivered_packets += 1

                    # Energy consumed during transmission
                    node.energy -= 0.01

                    if node.energy < 0:
                        node.energy = 0

    def display_statistics(self):

        print("\n========== Packet Transmission ==========\n")

        print("Packets Generated :", self.total_packets)
        print("Packets Delivered :", self.delivered_packets)
        print("Packet Loss :", self.total_packets - self.delivered_packets)