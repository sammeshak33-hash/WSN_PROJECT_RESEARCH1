import numpy as np
import matplotlib.pyplot as plt

from config import *

# Fix random seed
np.random.seed(RANDOM_SEED)


class SensorNode:
    def __init__(self, node_id, x, y, energy):
        self.id = node_id
        self.x = x
        self.y = y
        self.energy = energy

        self.is_cluster_head = False
        self.cluster_id = None
        self.trust = 1.0
        self.alive = True

        self.neighbors = []
        self.distance_to_bs = 0


def deploy_nodes():
    nodes = []

    for i in range(NUM_NODES):
        x = np.random.uniform(0, AREA_WIDTH)
        y = np.random.uniform(0, AREA_HEIGHT)

        node = SensorNode(
            i,
            x,
            y,
            INITIAL_ENERGY
        )

        nodes.append(node)

    return nodes


def plot_deployment(nodes):

    plt.figure(figsize=(8, 8))

    for node in nodes:
        plt.scatter(
            node.x,
            node.y,
            color="blue",
            s=25
        )

    plt.scatter(
        BASE_STATION_X,
        BASE_STATION_Y,
        color="red",
        marker="*",
        s=300,
        label="Base Station"
    )

    plt.xlim(0, AREA_WIDTH)
    plt.ylim(0, BASE_STATION_Y + 10)

    plt.title("Wireless Sensor Network Deployment")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    plt.grid(True)
    plt.legend()

    plt.show()