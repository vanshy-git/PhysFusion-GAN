import matplotlib.pyplot as plt
import numpy as np
import os
from . import config  # Import config from our own package

def visualize_processed_patch(patch_id=0):
    """
    Loads and visualizes a single processed patch pair (thermal and optical)
    from the 'data/processed' directory.
    
    Args:
        patch_id (int): The index of the patch to visualize (e.g., 0 for 'patch_0.npy').
    """
    print(f"Loading patch_{patch_id}.npy from processed data...")

    # --- Define file paths ---
    thermal_path = os.path.join(config.PROCESSED_DATA_DIR, "thermal", f"patch_{patch_id}.npy")
    optical_path = os.path.join(config.PROCESSED_DATA_DIR, "optical", f"patch_{patch_id}.npy")

    # --- Check if files exist ---
    if not os.path.exists(thermal_path) or not os.path.exists(optical_path):
        print(f"Error: Patch {patch_id} not found in '{config.PROCESSED_DATA_DIR}'")
        print("Please make sure you have run the 'run_pipeline.py' script first.")
        return

    # --- Load the numpy arrays ---
    try:
        thermal_patch = np.load(thermal_path)
        optical_patch = np.load(optical_path)
    except Exception as e:
        print(f"Error loading .npy files: {e}")
        return

    print(f"Loaded Optical patch shape: {optical_patch.shape}")
    print(f"Loaded Thermal patch shape: {thermal_patch.shape}")

    # --- Prepare images for plotting ---
    
    # Squeeze the thermal patch to remove the single-channel dim for plotting
    # (e.g., from (256, 256, 1) to (256, 256))
    if thermal_patch.shape[-1] == 1:
        thermal_patch_display = np.squeeze(thermal_patch, axis=-1)
    else:
        thermal_patch_display = thermal_patch
        
    # We assume the optical patch is already normalized [0,1] from our pipeline
    optical_patch_display = optical_patch

    # --- Create the visualization ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display Aligned (Blurry) Thermal
    axs[0].imshow(thermal_patch_display, cmap='hot')
    axs[0].set_title(f"Aligned Thermal (Input)\npatch_{patch_id}.npy")
    axs[0].axis('off')

    # Display Optical (Guide)
    axs[1].imshow(optical_patch_display)
    axs[1].set_title(f"Optical Guide (Input)\npatch_{patch_id}.npy")
    axs[1].axis('off')

    plt.suptitle(f"Verification of Processed Patch {patch_id}", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # This allows you to run this file directly from the terminal
    # to test your processed data.
    #
    # How to run:
    # 1. Open your terminal
    # 2. Make sure you are in the main 'PhysFusion-GAN' folder
    # 3. Run: python -m src.utils
    #
    # The '-m' flag tells Python to run the module as a script.
    
    visualize_processed_patch(0) # Tries to load the first patch by default