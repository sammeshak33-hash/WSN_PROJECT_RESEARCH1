import random


class PacketTransmission:

    def __init__(self, nodes):

        self.nodes = nodes

        self.total_packets = 0
        self.delivered_packets = 0
        self.packet_loss = 0

    # -------------------------------------------------
    # Generate Packets According to Attack Behaviour
    # -------------------------------------------------

    def generate_packets(self):

        for node in self.nodes:

            # Every node generates packets
            node.packet = 1

            # Initialize statistics
            node.transmission_count = 0
            node.forwarded_packets = 0
            node.packet_dropped = 0
            node.packet_delivery_ratio = 0.0

            # -----------------------------
            # Normal Node
            # -----------------------------

            if node.attack_type == "Normal":

                node.transmission_count = random.randint(20, 40)

                node.forwarded_packets = random.randint(
                    int(0.95 * node.transmission_count),
                    node.transmission_count
                )

                node.energy -= random.uniform(0.3, 0.8)

            # -----------------------------
            # Blackhole Attack
            # -----------------------------

            elif node.attack_type == "Blackhole":

                node.transmission_count = random.randint(20, 40)

                # Drops every packet
                node.forwarded_packets = 0

                node.energy -= random.uniform(0.2, 0.5)

            # -----------------------------
            # Grayhole Attack
            # -----------------------------

            elif node.attack_type == "Grayhole":

                node.transmission_count = random.randint(20, 40)

                node.forwarded_packets = random.randint(
                    int(0.40 * node.transmission_count),
                    int(0.60 * node.transmission_count)
                )

                node.energy -= random.uniform(0.5, 1.2)

            # -----------------------------
            # DoS Attack
            # -----------------------------

            elif node.attack_type == "DoS":

                node.transmission_count = random.randint(90, 120)

                node.forwarded_packets = random.randint(
                    int(0.80 * node.transmission_count),
                    node.transmission_count
                )

                node.energy -= random.uniform(10, 25)

            # -----------------------------
            # Energy Check
            # -----------------------------

            if node.energy < 0:
                node.energy = 0

            node.alive = node.energy > 0

            # -----------------------------
            # Packet Statistics
            # -----------------------------

            node.packet_dropped = (
                node.transmission_count -
                node.forwarded_packets
            )

            node.packet_delivery_ratio = (
                node.forwarded_packets /
                max(1, node.transmission_count)
            )

    # -------------------------------------------------
    # Packet Transmission Statistics
    # -------------------------------------------------

    def transmit_packets(self):

        self.total_packets = 0
        self.delivered_packets = 0

        for node in self.nodes:

            self.total_packets += node.transmission_count
            self.delivered_packets += node.forwarded_packets

        self.packet_loss = (
            self.total_packets -
            self.delivered_packets
        )

    # -------------------------------------------------
    # Display Overall Statistics
    # -------------------------------------------------

    def display_statistics(self):

        print("\n========== Packet Transmission ==========\n")

        print("Packets Generated :", self.total_packets)
        print("Packets Delivered :", self.delivered_packets)
        print("Packet Loss       :", self.packet_loss)

    # -------------------------------------------------
    # Display Sample Node Behaviour
    # -------------------------------------------------

    def display_sample_nodes(self, count=10):

        print("\n========== Sample Node Behaviour ==========\n")

        print(
            "Node  Attack       PDR    Forward  Drop  TxCount  Energy"
        )

        for node in self.nodes[:count]:

            print(
                f"{node.id:<5}"
                f"{node.attack_type:<13}"
                f"{node.packet_delivery_ratio:.2f}   "
                f"{node.forwarded_packets:<8}"
                f"{node.packet_dropped:<6}"
                f"{node.transmission_count:<9}"
                f"{node.energy:.2f}"
            )