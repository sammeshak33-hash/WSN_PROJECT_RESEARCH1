from config import *

# -------------------------------------------------------
# Calculate transmission energy
# -------------------------------------------------------
def calculate_transmit_energy(packet_size, distance):
    """
    Calculate energy required to transmit a packet.
    packet_size : bits
    distance    : meters
    """

    if distance < D0:
        energy = (packet_size * E_ELEC) + \
                 (packet_size * EPSILON_FS * (distance ** 2))
    else:
        energy = (packet_size * E_ELEC) + \
                 (packet_size * EPSILON_MP * (distance ** 4))

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
    """
    Deduct transmission energy from node.
    """

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
    """
    Deduct receiving energy from node.
    """

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
# Print energy status
# -------------------------------------------------------
def print_energy(node):

    print(
        f"Node {node.id} | Energy = {node.energy:.6f} J | Alive = {node.alive}"
    )

from config import INITIAL_ENERGY

def initialize_energy(nodes):
    """
    Initialize energy for all deployed nodes.
    """

    for node in nodes:
        node.energy = INITIAL_ENERGY
        node.alive = True    