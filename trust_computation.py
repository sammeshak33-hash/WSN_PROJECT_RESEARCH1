class TrustComputation:

    def __init__(self, nodes):

        self.nodes = nodes

    # ------------------------------------
    # Initialize Trust
    # ------------------------------------

    def initialize_trust(self):

        for node in self.nodes:

            node.direct_trust = 1.0
            node.indirect_trust = 1.0
            node.total_trust = 1.0

    # ------------------------------------
    # Direct Trust
    # ------------------------------------

    def calculate_direct_trust(self):

        for node in self.nodes:

            total_packets = (
                node.forwarded_packets +
                node.packet_dropped
            )

            if total_packets > 0:

                node.direct_trust = (
                    node.forwarded_packets /
                    total_packets
                )

            else:

                node.direct_trust = 0.0

    # ------------------------------------
    # Indirect Trust
    # ------------------------------------

    def calculate_indirect_trust(self):

        for node in self.nodes:

            neighbour_trust = []

            for neighbour in node.neighbors:

                neighbour_trust.append(
                    self.nodes[neighbour].direct_trust
                )

            if len(neighbour_trust) > 0:

                node.indirect_trust = (
                    sum(neighbour_trust)
                    /
                    len(neighbour_trust)
                )

            else:

                node.indirect_trust = node.direct_trust

    # ------------------------------------
    # Overall Trust
    # ------------------------------------

    def calculate_overall_trust(self):

        DIRECT_WEIGHT = 0.6
        INDIRECT_WEIGHT = 0.4

        for node in self.nodes:

            node.total_trust = (

                DIRECT_WEIGHT *
                node.direct_trust

                +

                INDIRECT_WEIGHT *
                node.indirect_trust

            )

    # ------------------------------------
    # Classification
    # ------------------------------------

    def classify_nodes(self):

        TRUST_THRESHOLD = 0.70

        trusted_nodes = []
        malicious_nodes = []

        for node in self.nodes:

            if node.total_trust >= TRUST_THRESHOLD:

                node.status = "Trusted"

                trusted_nodes.append(node.id)

            else:

                node.status = "Malicious"

                malicious_nodes.append(node.id)

        return trusted_nodes, malicious_nodes

    # ------------------------------------
    # Display Direct Trust
    # ------------------------------------

    def display_direct_trust(self):

        print("\n========== Direct Trust ==========\n")

        for node in self.nodes[:10]:

            print(
                f"Node {node.id:2d} : "
                f"{node.direct_trust:.3f}"
            )

    # ------------------------------------
    # Display Indirect Trust
    # ------------------------------------

    def display_indirect_trust(self):

        print("\n========== Indirect Trust ==========\n")

        for node in self.nodes[:10]:

            print(
                f"Node {node.id:2d} : "
                f"{node.indirect_trust:.3f}"
            )

    # ------------------------------------
    # Display Overall Trust
    # ------------------------------------

    def display_overall_trust(self):

        print("\n========== Overall Trust ==========\n")

        for node in self.nodes[:10]:

            print(
                f"Node {node.id:2d} : "
                f"{node.total_trust:.3f}"
            )

    # ------------------------------------
    # Display Classification
    # ------------------------------------

    def display_node_classification(
        self,
        trusted_nodes,
        malicious_nodes
    ):

        print("\n========== Node Classification ==========\n")

        print(
            f"Trusted Nodes ({len(trusted_nodes)}):"
        )
        print(trusted_nodes)

        print()

        print(
            f"Malicious Nodes ({len(malicious_nodes)}):"
        )
        print(malicious_nodes)          