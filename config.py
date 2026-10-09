"""
Project configuration.

Edit the values in this file instead of changing the main workflow.
"""

from pathlib import Path

SEED = 42

# Select the model used by the training workflow:
# "xgboost", "random_forest", or "extra_trees"
MODEL_NAME = "xgboost"

# Point-cloud input files. Paths are relative to the repository root.
# Keep the cluster IDs consistent with the train/test split used in the study.
TRAIN_FILES = [
    {"path": "data/train/mosteiro_SS_Junqueria_pc_train_1_b_sub.csv", "cluster": 1},
    {"path": "data/train/mosteiro_SS_Junqueria_pc_train_2_sub.csv", "cluster": 2},
    {"path": "data/train/mosteiro_SS_Junqueria_pc_train_3_sub.csv", "cluster": 3},
    {"path": "data/train/mosteiro_SS_Junqueria_pc_train_5_sub.csv", "cluster": 5},
    {"path": "data/train/mosteiro_SS_Junqueria_pc_train_6_sub.csv", "cluster": 6},
]

TEST_FILES = [
    {"path": "data/test/mosteiro_SS_Junqueria_pc_train_4_sub.csv", "cluster": 4},
    {"path": "data/test/mosteiro_SS_Junqueria_pc_train_7_sub.csv", "cluster": 7},
]

# Input files use semicolon-separated CSV and may use "//X" for the first column.
CSV_SEPARATOR = ";"
DROP_COLUMNS = ["Original cloud index"]

# KNN contextual feature parameters.
N_NEIGHBORS = 50

# Spatial tile size used to keep neighboring points in the same CV group.
TILE_SIZE = 0.25
N_CV_SPLITS = 5

# RandomizedSearchCV parameters.
N_RANDOM_SEARCH_ITERATIONS = 20
N_JOBS = 1

# Class labels used in the current dataset.
# Label 5 is mapped to background (0) as in the original main_2 workflow.
CLASS_REMAP = {5.0: 0.0}

CLASS_NAMES = {
    0: "Background",
    1: "Bacteria & Algae",
    2: "Lichens",
    3: "Mosses and Liverworts",
    4: "Plants",
}

# Colours used when exporting classified point clouds.
CLASS_COLORS = {
    0: "original_color",
    1: [0.0, 0.0, 0.0],
    2: [0.6510, 0.4549, 0.0157],
    3: [0.0, 1.0, 0.0],
    4: [0.0, 0.6667, 0.0],
    5: [0.4, 0.4, 0.4],
}

RESULTS_DIR = Path("results")
OUTPUTS_DIR = Path("outputs")

# Feature columns exported by CloudCompare and used by the workflow.
SPATIAL_FEATURES = [
    "Rf", "Gf", "Bf", "Intensity", "Nx", "Ny", "Nz",
    "Roughness (0.01)", "Mean curvature (0.01)",
    "Gaussian curvature (0.01)", "Normal change rate (0.01)",
    "Number of neighbors (r=0.01)", "Surface density (r=0.01)",
    "Omnivariance (0.01)", "Eigenentropy (0.01)", "Anisotropy (0.01)",
    "Planarity (0.01)", "Linearity (0.01)", "Sphericity (0.01)",
    "Verticality (0.01)",
    "Roughness (0.05)", "Mean curvature (0.05)",
    "Gaussian curvature (0.05)", "Normal change rate (0.05)",
    "Number of neighbors (r=0.05)", "Surface density (r=0.05)",
    "Omnivariance (0.05)", "Eigenentropy (0.05)", "Anisotropy (0.05)",
    "Planarity (0.05)", "Linearity (0.05)", "Sphericity (0.05)",
    "Verticality (0.05)",
]
