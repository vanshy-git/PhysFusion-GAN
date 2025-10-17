import torchvision.transforms as transforms

# Data is loaded as [0, 1], but Tanh output is [-1, 1]
# The formula (value * 2) - 1 maps [0, 1] to [-1, 1]

class NormalizeToTanh(object):
    """Normalizes a tensor in a sample dict from [0, 1] to [-1, 1]"""
    def __call__(self, sample):
        sample['LR_thermal'] = (sample['LR_thermal'] * 2) - 1
        sample['HR_optical'] = (sample['HR_optical'] * 2) - 1
        sample['HR_thermal'] = (sample['HR_thermal'] * 2) - 1
        return sample

# We can add data augmentation here later (e.g., random flips)
data_transform = transforms.Compose([
    NormalizeToTanh()
])