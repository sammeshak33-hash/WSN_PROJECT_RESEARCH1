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
from attack_injection import AttackInjection


class Simulation:

    def __init__(self):

        self.nodes = None
        self.cluster_heads = None
        self.clusters = None

        self.trust_model = None
        self.best_route = None

        self.packet_model = None
        self.feature_model = None

        self.trusted_nodes = None
        self.malicious_nodes = None

    def run(self):

        print("\n========== Simulation Started ==========\n")

        # ---------------------------------
        # Step 1 - Node Deployment
        # ---------------------------------

        self.nodes = deploy_nodes()

        print("Node Deployment Completed")

        # ---------------------------------
        # Step 1A - Attack Injection
        # ---------------------------------

        attack = AttackInjection(self.nodes)
        attack.assign_attack_nodes()

        print("Attack Injection Completed")

        # ---------------------------------
        # Step 2 - Neighbor Discovery
        # ---------------------------------

        discover_neighbors(self.nodes)

        print("Neighbor Discovery Completed")

        # ---------------------------------
        # Step 3 - Energy Initialization
        # ---------------------------------

        initialize_energy(self.nodes)

        print("Energy Initialization Completed")

        # ---------------------------------
        # Step 4 - IP-GJOA
        # ---------------------------------

        optimizer = IPGJOA(len(self.nodes))

        optimizer.initialize_population()

        fitness = FitnessCalculator(self.nodes)

        best_solution, best_fitness, history = optimizer.optimize(fitness)

        self.cluster_heads = best_solution

        print("Ip-GJOA Completed")

        # ---------------------------------
        # Step 5 - Cluster Formation
        # ---------------------------------

        cluster = ClusterFormation(self.nodes)

        self.clusters = cluster.form_clusters(self.cluster_heads)

        print("Cluster Formation Completed")

        # ---------------------------------
        # Step 6 - Packet Transmission
        # ---------------------------------

        packet = PacketTransmission(self.nodes)

        packet.generate_packets()

        # Your new PacketTransmission uses attack_type
        packet.transmit_packets()

        packet.display_statistics()

        packet.display_sample_nodes()

        self.packet_model = packet

        print("Packet Transmission Completed")

        # ---------------------------------
        # Step 7 - Trust Computation
        # ---------------------------------

        trust = TrustComputation(self.nodes)

        trust.initialize_trust()

        trust.calculate_direct_trust()

        trust.calculate_indirect_trust()

        trust.calculate_overall_trust()

        self.trusted_nodes, self.malicious_nodes = trust.classify_nodes()

        self.trust_model = trust

        print("Trust Computation Completed")

        # ---------------------------------
        # Step 8 - SOA Routing
        # ---------------------------------

        routing = SOARouting(
            self.nodes,
            self.clusters
        )

        routing.initialize_routes()

        fitness_values = routing.calculate_route_fitness()

        routing.select_male_female_snakes(
            fitness_values
        )

        best_route, best_fitness, history = routing.optimize_routes()

        self.best_route = best_route

        print("SOA Routing Completed")

        # ---------------------------------
        # Step 9 - Feature Extraction
        # ---------------------------------

        feature = FeatureExtraction(self.nodes)

        feature.extract_features()

        feature.save_dataset()

        self.feature_model = feature

        print("Feature Extraction Completed")
        print("Dataset Saved Successfully")

        print("\n========== Simulation Finished ==========\n")

        return {

            "nodes": self.nodes,

            "cluster_heads": self.cluster_heads,

            "clusters": self.clusters,

            "trust_model": self.trust_model,

            "trusted_nodes": self.trusted_nodes,

            "malicious_nodes": self.malicious_nodes,

            "best_route": self.best_route,

            "feature_model": self.feature_model

        }