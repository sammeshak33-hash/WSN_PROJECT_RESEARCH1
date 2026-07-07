# ------------------------------
# Wireless Sensor Network Configuration
# ------------------------------

# Simulation Area
AREA_WIDTH = 100
AREA_HEIGHT = 100

# Number of Sensor Nodes
NUM_NODES = 100

# Initial Energy (Joule)
INITIAL_ENERGY = 1.0

# Communication Range (meters)
COMMUNICATION_RANGE = 25

# Base Station Coordinates
BASE_STATION_X = 50
BASE_STATION_Y = 120

# Random Seed (for reproducibility)
RANDOM_SEED = 42

# ---------------------------------------------------
# Energy Model Parameters
# ---------------------------------------------------

# Energy consumed by radio electronics (J/bit)
E_ELEC = 50e-9

# Free Space Amplifier (J/bit/m²)
EPSILON_FS = 10e-12

# Multi-path Amplifier (J/bit/m⁴)
EPSILON_MP = 0.0013e-12

# Packet Size (bits)
PACKET_SIZE = 4000

# Threshold Distance
D0 = (EPSILON_FS / EPSILON_MP) ** 0.5

POPULATION_SIZE = 20
NUM_CLUSTER_HEADS = 5
MAX_ITERATIONS = 50

ALPHA = 0.4
BETA = 0.35
GAMMA = 0.25