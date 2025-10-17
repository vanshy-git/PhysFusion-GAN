import os
import numpy as np
import cv2
import rasterio  # <-- Our new library
from glob import glob
from tqdm import tqdm
from . import config

def get_rgb(all_bands_image):
    """
    Extracts and stacks the Red, Green, and Blue bands to create an RGB image.
    (This function is correct and does not need to change)
    """
    # Bands are 0-indexed: Band 4=[3], Band 3=[2], Band 2=[1]
    red = all_bands_image[3, :, :]
    green = all_bands_image[2, :, :]
    blue = all_bands_image[1, :, :]
    
    rgb = np.dstack((red, green, blue))
    
    # Normalize for processing
    min_val, max_val = np.min(rgb), np.max(rgb)
    if max_val > min_val:
        rgb = (rgb - min_val) / (max_val - min_val)
        
    return rgb.astype(np.float32)

def get_thermal(all_bands_image):
    """
    Extracts the Thermal Infrared band (Band 10).
    (This function is correct and does not need to change)
    """
    # Band 10 is at index 9
    thermal = all_bands_image[9, :, :]
    
    # Normalize
    min_val, max_val = np.min(thermal), np.max(thermal)
    if max_val > min_val:
        thermal = (thermal - min_val) / (max_val - min_val)
        
    return thermal.astype(np.float32)

def create_and_save_patches():
    """
    Loads .tif data from the benchmark, processes it into aligned pairs,
    and saves them as patches.
    """
    # --- THIS IS THE KEY CHANGE ---
    # Search recursively for all .tif files in the dataset folder.
    # The `**` and `recursive=True` will find files in any subdirectories.
    print(f"Searching for .tif files in: {config.DATASET_FOLDER}")
    data_files = glob(os.path.join(config.DATASET_FOLDER, "**", "*.tif"), recursive=True)
    # --- END OF CHANGE ---

    if not data_files:
        print(f"Error: No .tif files found in {config.DATASET_FOLDER}.")
        print("Please check the folder name in src/config.py and ensure you have unzipped the data.")
        return

    print(f"Found {len(data_files)} .tif files to process.")

    # Create output directories
    thermal_dir = os.path.join(config.PROCESSED_DATA_DIR, "thermal")
    optical_dir = os.path.join(config.PROCESSED_DATA_DIR, "optical")
    os.makedirs(thermal_dir, exist_ok=True)
    os.makedirs(optical_dir, exist_ok=True)

    patch_count = 0
    for file_path in tqdm(data_files, desc="Processing .tif files"):
        try:
            # --- THIS IS THE KEY CHANGE ---
            # Open the .tif file and read all 11 bands
            with rasterio.open(file_path) as src:
                all_bands = src.read() # This loads as (bands, height, width)
            # --- END OF CHANGE ---

            # Check if we have the expected 11 bands
            if all_bands.shape[0] != 11:
                print(f"Skipping {file_path}: Expected 11 bands, but found {all_bands.shape[0]}")
                continue
            
            # Separate into high-res optical and low-res thermal
            hr_optical = get_rgb(all_bands)
            lr_thermal = get_thermal(all_bands)

            # Align: Upscale the thermal image to match the optical image's size
            target_size = (hr_optical.shape[1], hr_optical.shape[0])
            lr_aligned = cv2.resize(lr_thermal, target_size, interpolation=cv2.INTER_CUBIC)
            
            # Add a channel dimension for consistency
            lr_aligned = lr_aligned[:, :, np.newaxis]

            # Save the processed pair (still as .npy, which is best for training)
            np.save(os.path.join(thermal_dir, f"patch_{patch_count}.npy"), lr_aligned)
            np.save(os.path.join(optical_dir, f"patch_{patch_count}.npy"), hr_optical)
            patch_count += 1
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            
    print(f"\nSuccessfully created and saved {patch_count} patch pairs.")