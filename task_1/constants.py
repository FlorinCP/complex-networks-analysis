import os

DATA_DIR = "data/A3_synthetic_networks"
RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_NODES = 300
N_BLOCKS = 5
PRS = 0.02
PRR_VALUES = [0.0, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
