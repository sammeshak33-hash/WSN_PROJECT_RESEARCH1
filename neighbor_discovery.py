import math
from config import *

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def discover_neighbors(nodes):
    for node in nodes:

        node.neighbors = []

        node.distance_to_bs = euclidean_distance(
            node.x,
            node.y,
            BASE_STATION_X,
            BASE_STATION_Y
        )

        for other in nodes:

            if node.id == other.id:
                continue

            d = euclidean_distance(
                node.x,
                node.y,
                other.x,
                other.y
            )

            if d <= COMMUNICATION_RANGE:
                node.neighbors.append(other.id)

    return nodes