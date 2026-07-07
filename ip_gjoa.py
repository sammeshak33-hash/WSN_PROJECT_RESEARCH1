import math
import random
from config import *


class IPGJOA:

    def __init__(self, total_nodes):
        self.total_nodes = total_nodes
        self.population_size = POPULATION_SIZE
        self.num_cluster_heads = NUM_CLUSTER_HEADS
        self.population = []

    def initialize_population(self):
        """
        Generate random candidate Cluster Head solutions.
        """

        self.population = []

        for i in range(self.population_size):

            candidate = random.sample(
                range(self.total_nodes),
                self.num_cluster_heads
            )

            candidate.sort()

            self.population.append(candidate)

        return self.population

    def display_population(self):

        print("\n========== Initial Population ==========\n")

        for i, solution in enumerate(self.population):
            print(f"Solution {i+1:2d} : {solution}")

        print("\n========================================")

    def hunting_energy(self, iteration):
        L1 = 1

        M = (
            0.05
            * L1
            * math.exp(
                -2 * ((iteration / MAX_ITERATIONS) ** 2)
            )
        )

        return M 
       
    def select_male_female_jackals(self, fitness_values):
        """
        Select the best (Male) and second-best (Female)
        solutions based on fitness.
        """

        ranked = list(zip(self.population, fitness_values))

        ranked.sort(
            key=lambda x: x[1],
            reverse=True
        )

        male = ranked[0]
        female = ranked[1]

        return male, female
    

    def update_position(self, current_solution, male_solution, female_solution):
        """
        Update a candidate solution by moving it
        toward the Male and Female Jackals.
        """

        new_solution = []

        for i in range(self.num_cluster_heads):

            current = current_solution[i]
            male = male_solution[i]
            female = female_solution[i]

            new_position = round(
                (current + male + female) / 3
            )

            # Keep node ID within valid range
            new_position = max(
                0,
                min(new_position, self.total_nodes - 1)
            )

            new_solution.append(new_position)

        # Remove duplicate node IDs
        new_solution = list(set(new_solution))

        # Repair solution if duplicates were removed
        while len(new_solution) < self.num_cluster_heads:

            candidate = random.randint(
                0,
                self.total_nodes - 1
            )

            if candidate not in new_solution:
                new_solution.append(candidate)

        new_solution.sort()

        return new_solution
    def exploration_phase(self, solution):
        """
        Perform exploration by randomly replacing
        one Cluster Head with another valid node.
        """

        explored_solution = solution.copy()

        # Select one Cluster Head randomly
        index = random.randint(
            0,
            self.num_cluster_heads - 1
        )

        # Generate a new node
        new_node = random.randint(
            0,
            self.total_nodes - 1
        )

        # Avoid duplicate Cluster Heads
        while new_node in explored_solution:
            new_node = random.randint(
                0,
                self.total_nodes - 1
            )

        explored_solution[index] = new_node

        explored_solution.sort()

        return explored_solution
    def exploitation_phase(self, solution):
        """
        Perform exploitation by making a small local
        adjustment to one Cluster Head.
        """

        exploited_solution = solution.copy()

        index = random.randint(
            0,
            self.num_cluster_heads - 1
        )

        step = random.choice([-1, 1])

        new_node = exploited_solution[index] + step

        new_node = max(
            0,
            min(new_node, self.total_nodes - 1)
        )

        while new_node in exploited_solution:
            new_node = random.randint(
                0,
                self.total_nodes - 1
            )

        exploited_solution[index] = new_node

        exploited_solution.sort()

        return exploited_solution
    def optimize(self, fitness_calculator):
        """
        Complete IP-GJOA optimization process.
        """

        best_solution = None
        best_fitness = -1
        fitness_history = []

        for iteration in range(MAX_ITERATIONS):

            print(f"\nIteration {iteration + 1}")

            

            # ----------------------------
            # Evaluate Fitness
            # ----------------------------
            fitness_values = []

            for solution in self.population:
                fitness = fitness_calculator.calculate_fitness(solution)
                fitness_values.append(fitness)

            # ----------------------------
            # Select Male & Female
            # ----------------------------
            male, female = self.select_male_female_jackals(
                fitness_values
            )

            # ----------------------------
            # Save Best Solution
            # ----------------------------
            if male[1] > best_fitness:
                best_fitness = male[1]
                best_solution = male[0].copy()

            fitness_history.append(best_fitness)

            print("Best Fitness :", round(best_fitness, 6))

            # ----------------------------
            # Update Population
            # ----------------------------
            new_population = []

            for solution in self.population:

                updated = self.update_position(
                    solution,
                    male[0],
                    female[0]
                )

                explored = self.exploration_phase(updated)

                exploited = self.exploitation_phase(explored)

                new_population.append(exploited)

                

            self.population = new_population

        return best_solution, best_fitness, fitness_history