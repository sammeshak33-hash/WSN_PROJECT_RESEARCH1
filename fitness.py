from config import *


class FitnessCalculator:

    def __init__(self, nodes):
        self.nodes = nodes

    def calculate_average_energy(self, solution):

        total_energy = 0

        for node_id in solution:
            total_energy += self.nodes[node_id].energy

        return total_energy / len(solution)

    def calculate_average_trust(self, solution):

        total_trust = 0

        for node_id in solution:
            total_trust += self.nodes[node_id].trust

        return total_trust / len(solution)

    def calculate_path_distance(self, solution):

        total_distance = 0

        for node_id in solution:
            total_distance += self.nodes[node_id].distance_to_bs

        return total_distance
    def calculate_fitness(self, solution):

        avg_energy = self.calculate_average_energy(solution)

        avg_trust = self.calculate_average_trust(solution)

        path_distance = self.calculate_path_distance(solution)

        fitness = (
            ALPHA * (1 / path_distance)
            + BETA * avg_energy
            + GAMMA * avg_trust
        )

        return fitness