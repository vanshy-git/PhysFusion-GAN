from src import data_processing, config

def main():
    """Runs the full data processing pipeline for the benchmark dataset."""
    print("--- Starting PhysFusion-GAN Data Pipeline ---")
    
    # The new function handles everything: finding, loading, processing, and saving.
    data_processing.create_and_save_patches()
    
    print("\n--- Pipeline Finished Successfully! ---")
    print(f"Processed data saved to: {config.PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()