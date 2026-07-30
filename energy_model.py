from config import *

# -------------------------------------------------------
# Calculate transmission energy
# -------------------------------------------------------
def calculate_transmit_energy(packet_size, distance):
    """
    Calculate energy required to transmit a packet.
    """

    if distance < D0:
        energy = (
            packet_size * E_ELEC
            + packet_size * EPSILON_FS * (distance ** 2)
        )
    else:
        energy = (
            packet_size * E_ELEC
            + packet_size * EPSILON_MP * (distance ** 4)
        )

    return energy


# -------------------------------------------------------
# Calculate reception energy
# -------------------------------------------------------
def calculate_receive_energy(packet_size):
    """
    Energy required to receive a packet.
    """

    return packet_size * E_ELEC


# -------------------------------------------------------
# Transmit packet
# -------------------------------------------------------
def transmit_packet(node, packet_size, distance):

    energy = calculate_transmit_energy(packet_size, distance)

    node.energy -= energy

    if node.energy < 0:
        node.energy = 0

    update_residual_energy(node)

    return energy


# -------------------------------------------------------
# Receive packet
# -------------------------------------------------------
def receive_packet(node, packet_size):

    energy = calculate_receive_energy(packet_size)

    node.energy -= energy

    if node.energy < 0:
        node.energy = 0

    update_residual_energy(node)

    return energy


# -------------------------------------------------------
# Update node status
# -------------------------------------------------------
def update_residual_energy(node):

    if node.energy <= 0:
        node.alive = False
    else:
        node.alive = True


# -------------------------------------------------------
# Print energy
# -------------------------------------------------------
def print_energy(node):

    print(
        f"Node {node.id} | "
        f"Energy = {node.energy:.6f} J | "
        f"Alive = {node.alive}"
    )


# -------------------------------------------------------
# Initialize Energy
# -------------------------------------------------------
from config import INITIAL_ENERGY
import random


def initialize_energy(nodes):
    """
    Initialize node energy according to attack behaviour.
    """

    for node in nodes:

        attack = getattr(node, "attack_type", "Normal")

        if attack == "DoS":
            # DoS nodes start with lower energy
            node.energy = random.uniform(10, 25)

        else:
            node.energy = INITIAL_ENERGY

        node.alive = True