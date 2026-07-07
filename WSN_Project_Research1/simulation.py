from deployment import deploy_nodes
from neighbor_discovery import discover_neighbors
from energy_model import initialize_energy

from ip_gjoa import IPGJOA
from fitness import FitnessCalculator
from cluster_formation import ClusterFormation
from trust_computation import TrustComputation

from soa_routing import SOARouting
from packet_transmission import PacketTransmission
from feature_extraction import FeatureExtraction

from dataset_validation import DatasetValidation
from preprocessing import Preprocessing
class Simulation:

    def __init__(self):
            self.nodes = None

            self.cluster_heads = None

            self.clusters = None

            self.trust_model = None

            self.best_route = None

            self.packet_model = None

            self.feature_model = None

    def run(self):

        print("\n========== Simulation Started ==========\n")

        # Step 1 - Node Deployment
        self.nodes = deploy_nodes()
        print("Node Deployment Completed")

        # Step 2 - Neighbor Discovery
        discover_neighbors(self.nodes)
        print("Neighbor Discovery Completed")

        # Step 3 - Energy Initialization
        initialize_energy(self.nodes)
        print("Energy Initialization Completed")

        # ----------------------------
        # Step 4 - Ip-GJOA
        # ----------------------------
        optimizer = IPGJOA(len(self.nodes))

        optimizer.initialize_population()

        fitness = FitnessCalculator(self.nodes)

        best_solution, best_fitness, history = optimizer.optimize(fitness)

        self.cluster_heads = best_solution

        print("Ip-GJOA Completed")

        # ----------------------------
        # Step 5 - Cluster Formation
        # ----------------------------
        cluster = ClusterFormation(self.nodes)

        self.clusters = cluster.form_clusters(self.cluster_heads)

        print("Cluster Formation Completed")

        # ----------------------------
        # Step 6 - Trust Computation
        # ----------------------------
        trust = TrustComputation(self.nodes)

        trust.initialize_trust()

        trust.calculate_direct_trust()

        trust.calculate_indirect_trust()

        trust.calculate_overall_trust()

        self.trusted_nodes, self.malicious_nodes = trust.classify_nodes()

        self.trust_model = trust

        print("Trust Computation Completed")

        # ----------------------------
        # Step 7 - SOA Routing
        # ----------------------------

        routing = SOARouting(self.nodes, self.clusters)

        routing.initialize_routes()

        fitness_values = routing.calculate_route_fitness()

        routing.select_male_female_snakes(fitness_values)

        best_route, best_fitness, history = routing.optimize_routes()

        self.best_route = best_route

        print("SOA Routing Completed")

        # ----------------------------
        # Step 8 - Packet Transmission
        # ----------------------------

        packet = PacketTransmission(self.nodes)

        packet.generate_packets()

        packet.transmit_packets(self.trusted_nodes)

        self.packet_model = packet

        print("Packet Transmission Completed")

        # ----------------------------
        # Step 9 - Feature Extraction
        # ----------------------------

        feature = FeatureExtraction(self.nodes)

        feature.extract_features()

        feature.save_dataset()

        self.feature_model = feature

        print("Feature Extraction Completed")

        print("Dataset Saved Successfully")

        # ---------------------------------------
        # Step 10 - Dataset Validation
        # ---------------------------------------

        validator = DatasetValidation("network_dataset.csv")

        validator.dataset_info()
        validator.attack_distribution()
        validator.missing_values()
        validator.duplicate_rows()

        print("Dataset Validation Completed")

        # ---------------------------------------
        # Step 11 - Dataset Preprocessing
        # ---------------------------------------

        prep = Preprocessing("network_dataset.csv")

        prep.separate_features()
        prep.encode_labels()
        prep.normalize()
        prep.split_dataset()

        # Reshape Dataset for PRCNN

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

        print("\nDataset Reshaped Successfully")
        print("Training Shape :", prep.X_train.shape)
        print("Testing Shape  :", prep.X_test.shape)

        self.prep = prep

        print("Preprocessing Completed")

        print("\n========== Simulation Finished ==========\n")

        return {
            "nodes": self.nodes,
            "cluster_heads": self.cluster_heads,
            "clusters": self.clusters,
            "trust_model": self.trust_model,
            "trusted_nodes": self.trusted_nodes,
            "malicious_nodes": self.malicious_nodes,
            "best_route": self.best_route,
            "feature_model": self.feature_model,
            "prep": self.prep
        }