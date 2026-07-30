import random


class PacketTransmission:

    def __init__(self, nodes):

        self.nodes = nodes

        self.total_packets = 0
        self.delivered_packets = 0

    def generate_packets(self):
        """
        Generate packets according to each node's attack behaviour.
        """

        for node in self.nodes:

            # Initialize packet statistics
            node.packet = 1
            node.transmission_count = 0
            node.packet_delivered = 0
            node.packet_dropped = 0
            node.packet_delivery_ratio = 0.0

            # -------------------------
            # Normal Node
            # -------------------------
            if node.attack_type == "Normal":

                node.transmission_count = random.randint(20, 40)

                node.packet_delivered = random.randint(
                    int(0.90 * node.transmission_count),
                    node.transmission_count
                )

            # -------------------------
            # Blackhole Node
            # -------------------------
            elif node.attack_type == "Blackhole":

                node.transmission_count = random.randint(20, 40)

                # Drops every packet
                node.packet_delivered = 0

            # -------------------------
            # Grayhole Node
            # -------------------------
            elif node.attack_type == "Grayhole":

                node.transmission_count = random.randint(20, 40)

                node.packet_delivered = random.randint(
                    int(0.40 * node.transmission_count),
                    int(0.60 * node.transmission_count)
                )

            # -------------------------
            # DoS Node
            # -------------------------
            elif node.attack_type == "DoS":

                node.transmission_count = random.randint(90, 120)

                node.packet_delivered = random.randint(
                    int(0.80 * node.transmission_count),
                    node.transmission_count
                )

                # Heavy energy consumption
                node.energy -= random.uniform(10, 25)

                if node.energy < 0:
                    node.energy = 0

            # -------------------------
            # Statistics
            # -------------------------
            node.packet_dropped = (
                node.transmission_count -
                node.packet_delivered
            )

            node.packet_delivery_ratio = (
                node.packet_delivered /
                node.transmission_count
            )

    def transmit_packets(self, trusted_nodes):
        """
        Update overall transmission statistics.
        """

        self.total_packets = 0
        self.delivered_packets = 0

        for node in self.nodes:

            self.total_packets += node.transmission_count
            self.delivered_packets += node.packet_delivered

        self.packet_loss = (
            self.total_packets -
            self.delivered_packets
        )

    def display_statistics(self):

        print("\n========== Packet Transmission ==========\n")

        print("Packets Generated :", self.total_packets)
        print("Packets Delivered :", self.delivered_packets)
        print("Packet Loss :", self.packet_loss)