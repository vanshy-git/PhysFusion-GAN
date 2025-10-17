import os
import numpy as np
import cv2
import rasterio
from glob import glob
from tqdm import tqdm
from . import config

def get_rgb(all_bands_image):
    """Extracts and normalizes the RGB bands (B4, B3, B2)."""
    red = all_bands_image[3, :, :]
    green = all_bands_image[2, :, :]
    blue = all_bands_image[1, :, :]
    
    rgb = np.dstack((red, green, blue))
    
    # Normalize to [0, 1]
    min_val, max_val = np.min(rgb), np.max(rgb)
    if max_val > min_val:
        rgb = (rgb - min_val) / (max_val - min_val)
        
    return rgb.astype(np.float32)

def get_thermal(all_bands_image):
    """Extracts and normalizes the Thermal band (B10)."""
    # Band 10 is at index 9
    thermal = all_bands_image[9, :, :]
    
    # Normalize to [0, 1]
    min_val, max_val = np.min(thermal), np.max(thermal)
    if max_val > min_val:
        thermal = (thermal - min_val) / (max_val - min_val)
        
    return thermal.astype(np.float32)

def create_and_save_patches():
    """
    Loads .tif data, processes it into (input, guide, ground_truth) tuples,
    and saves them as patches.
    """
    print(f"Searching for .tif files in: {config.DATASET_FOLDER}")
    data_files = glob(os.path.join(config.DATASET_FOLDER, "**", "*.tif"), recursive=True)

    if not data_files:
        print(f"Error: No .tif files found in {config.DATASET_FOLDER}.")
        return
    print(f"Found {len(data_files)} .tif files to process.")

    # --- Create output directories for all three components ---
    lr_thermal_dir = os.path.join(config.PROCESSED_DATA_DIR, "lr_thermal")
    hr_optical_dir = os.path.join(config.PROCESSED_DATA_DIR, "hr_optical")
    hr_thermal_dir = os.path.join(config.PROCESSED_DATA_DIR, "hr_thermal")
    
    os.makedirs(lr_thermal_dir, exist_ok=True, mode=0o777)
    os.makedirs(hr_optical_dir, exist_ok=True, mode=0o777)
    os.makedirs(hr_thermal_dir, exist_ok=True, mode=0o777)

    patch_count = 0
    patch_size = config.PATCH_SIZE
    scale_factor = config.SCALE_FACTOR

    for file_path in tqdm(data_files, desc="Processing .tif files"):
        try:
            with rasterio.open(file_path) as src:
                all_bands = src.read()
            if all_bands.shape[0] != 11:
                continue

            # --- 1. Get Optical Guide and Thermal Ground Truth ---
            hr_optical = get_rgb(all_bands)
            hr_thermal = get_thermal(all_bands) # This is our ground truth!

            # --- 2. Create the Low-Resolution Thermal Input ---
            # Downsample the ground truth by 4x
            lr_shape = (hr_thermal.shape[1] // scale_factor, hr_thermal.shape[0] // scale_factor)
            lr_temp = cv2.resize(hr_thermal, lr_shape, interpolation=cv2.INTER_AREA)
            
            # Upsample it back using simple interpolation (to make it blurry)
            hr_shape = (hr_thermal.shape[1], hr_thermal.shape[0])
            lr_thermal = cv2.resize(lr_temp, hr_shape, interpolation=cv2.INTER_CUBIC)

            # --- 3. Chip and Save ---
            for y in range(0, hr_optical.shape[0] - patch_size + 1, patch_size):
                for x in range(0, hr_optical.shape[1] - patch_size + 1, patch_size):
                    
                    # Extract patches
                    patch_hr_opt = hr_optical[y:y+patch_size, x:x+patch_size, :]
                    patch_hr_th = hr_thermal[y:y+patch_size, x:x+patch_size]
                    patch_lr_th = lr_thermal[y:y+patch_size, x:x+patch_size]
                    
                    # Add channel dimension
                    patch_hr_th = patch_hr_th[:, :, np.newaxis]
                    patch_lr_th = patch_lr_th[:, :, np.newaxis]

                    # Save the three corresponding files
                    np.save(os.path.join(lr_thermal_dir, f"patch_{patch_count}.npy"), patch_lr_th)
                    np.save(os.path.join(hr_optical_dir, f"patch_{patch_count}.npy"), patch_hr_opt)
                    np.save(os.path.join(hr_thermal_dir, f"patch_{patch_count}.npy"), patch_hr_th)
                    patch_count += 1
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            
    print(f"\nSuccessfully created and saved {patch_count} patch triplets.")