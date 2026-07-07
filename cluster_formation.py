import math


class ClusterFormation:

    def __init__(self, nodes):
        self.nodes = nodes
        self.clusters = {}

    def form_clusters(self, cluster_heads):

        self.clusters = {}

        # Create empty clusters
        for ch in cluster_heads:
            self.clusters[ch] = []

        # Assign every node
        for node in self.nodes:

            nearest_ch = None
            minimum_distance = float("inf")

            for ch in cluster_heads:

                ch_node = self.nodes[ch]

                distance = math.sqrt(
                    (node.x - ch_node.x) ** 2 +
                    (node.y - ch_node.y) ** 2
                )

                if distance < minimum_distance:
                    minimum_distance = distance
                    nearest_ch = ch

            self.clusters[nearest_ch].append(node.id)

        return self.clusters

    def display_clusters(self):

        print("\n========== Cluster Formation ==========\n")

        for ch, members in self.clusters.items():

            print(f"Cluster Head : {ch}")
            print(f"Members ({len(members)}):")
            print(members)
            print("--------------------------------")