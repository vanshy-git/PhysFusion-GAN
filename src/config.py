import os

# --- Project Root ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# --- New Dataset Configuration ---
DATASET_FOLDER = os.path.join(RAW_DATA_DIR, "ssl4eo_l_oli_tirs_toa_benchmark", "ssl4eo_l_oli_tirs_toa_benchmark")

# --- Processing Parameters ---
PATCH_SIZE = 256 # The size of the final training patches.
SCALE_FACTOR = 4 # We will create a 4x super-resolution task.