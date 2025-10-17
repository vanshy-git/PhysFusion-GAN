import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from . import config, transforms

class SuperResDataset(Dataset):
    """Custom Dataset for loading (LR_thermal, HR_optical, HR_thermal) patches."""
    def __init__(self, data_dir, transform=None):
        self.lr_thermal_dir = os.path.join(data_dir, "lr_thermal")
        self.hr_optical_dir = os.path.join(data_dir, "hr_optical")
        self.hr_thermal_dir = os.path.join(data_dir, "hr_thermal")
        
        # Get a sorted list of all patch filenames
        self.lr_thermal_files = sorted(glob(os.path.join(self.lr_thermal_dir, "*.npy")))
        self.hr_optical_files = sorted(glob(os.path.join(self.hr_optical_dir, "*.npy")))
        self.hr_thermal_files = sorted(glob(os.path.join(self.hr_thermal_dir, "*.npy")))
        
        self.transform = transform
        
        # Sanity check
        if not (len(self.lr_thermal_files) == len(self.hr_optical_files) == len(self.hr_thermal_files)):
            raise ValueError("Mismatch in number of patches between directories!")
        if len(self.lr_thermal_files) == 0:
            raise FileNotFoundError(f"No .npy files found in {data_dir}. Did you run run_pipeline.py?")
            
        print(f"Found {len(self.lr_thermal_files)} patch triplets.")

    def __len__(self):
        return len(self.lr_thermal_files)

    def __getitem__(self, idx):
        # Load the numpy arrays
        lr_thermal = np.load(self.lr_thermal_files[idx]).astype(np.float32)
        hr_optical = np.load(self.hr_optical_files[idx]).astype(np.float32)
        hr_thermal = np.load(self.hr_thermal_files[idx]).astype(np.float32)
        
        # Transpose from (H, W, C) to (C, H, W) for PyTorch
        lr_thermal = np.transpose(lr_thermal, (2, 0, 1))
        hr_optical = np.transpose(hr_optical, (2, 0, 1))
        hr_thermal = np.transpose(hr_thermal, (2, 0, 1))
        
        # Convert to PyTorch tensors
        sample = {
            'LR_thermal': torch.from_numpy(lr_thermal),
            'HR_optical': torch.from_numpy(hr_optical),
            'HR_thermal': torch.from_numpy(hr_thermal)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    # This block allows us to test the dataloader directly
    from torch.utils.data import DataLoader
    
    print("Testing Dataloader...")
    try:
        dataset = SuperResDataset(
            data_dir=config.PROCESSED_DATA_DIR,
            transform=transforms.data_transform # Apply the transform
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample 0 shapes:")
            print(f"LR_thermal: {sample['LR_thermal'].shape} -> Min: {sample['LR_thermal'].min():.2f}, Max: {sample['LR_thermal'].max():.2f}")
            print(f"HR_optical: {sample['HR_optical'].shape} -> Min: {sample['HR_optical'].min():.2f}, Max: {sample['HR_optical'].max():.2f}")
            print(f"HR_thermal: {sample['HR_thermal'].shape} -> Min: {sample['HR_thermal'].min():.2f}, Max: {sample['HR_thermal'].max():.2f}")
            print("(Values should be in the [-1, 1] range)")
            
            loader = DataLoader(dataset, batch_size=4, shuffle=True)
            batch = next(iter(loader))
            print(f"\nBatch shapes:")
            print(f"LR_thermal batch: {batch['LR_thermal'].shape}")
            print(f"HR_optical batch: {batch['HR_optical'].shape}")
            print(f"HR_thermal batch: {batch['HR_thermal'].shape}")
            print("\nDataloader test successful!")
        else:
            print("Dataset is empty. Run 'run_pipeline.py' first.")
    except Exception as e:
        print(f"\nDataloader test FAILED: {e}")