import os

# --- Project Root ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# --- New Dataset Configuration ---
# Point to the folder you just unzipped. You may need to adjust the folder name.
DATASET_FOLDER = os.path.join(RAW_DATA_DIR, "ssl4eo_l_oli_tirs_toa_benchmark", "ssl4eo_l_oli_tirs_toa_benchmark")

# --- Processing Parameters ---
# The task is to super-resolve Band 10 and Band 11[cite: 10]. We'll focus on Band 10.
# The optical bands have a 30m resolution, and thermal is 100m[cite: 8, 9].
# This is a ~3.33x difference, making a 4x super-resolution task ideal[cite: 12].

PATCH_SIZE = 256 # The size of the final training patches.

# Normalization will be handled based on the data's own min/max values.