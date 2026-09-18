import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Mapping of GTSRB class IDs to human-readable names
GTSRB_CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}
def get_transforms(is_train=True):
    """
    Returns torchvision transforms for preprocessing and augmentation.
    
    Args:
        is_train (bool): If True, includes data augmentation (rotation, color jitter, affine).
                         If False, only includes resizing, tensor conversion, and normalization.
    """
    # GTSRB native resolution is roughly around 32x32 to 100x100. We standardize to 32x32.
    base_transforms = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # Normalization values approximate those used for GTSRB in standard literature
        transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
    ]
    
    if is_train:
        # Augmentations to simulate real-world conditions (varying angles, lighting, position)
        aug_transforms = [
            transforms.RandomRotation(15), # Slight rotations
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Lighting changes
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Slight shifts
        ]
        return transforms.Compose(aug_transforms + base_transforms)
    else:
        return transforms.Compose(base_transforms)

def get_dataloaders(data_dir, batch_size=64, val_split=0.2):
    """
    Downloads/loads GTSRB dataset and creates train, validation, and test dataloaders.
    
    Args:
        data_dir (str): Path to store/find the dataset.
        batch_size (int): Number of images per batch.
        val_split (float): Fraction of training data to use for validation.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Load full training dataset with training transforms
    full_train_dataset = datasets.GTSRB(
        root=data_dir,
        split='train',
        transform=get_transforms(is_train=True),
        download=True
    )
    
    # Calculate lengths for train/val split
    val_len = int(len(full_train_dataset) * val_split)
    train_len = len(full_train_dataset) - val_len
    # Generate deterministic indices for splitting
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_train_dataset), generator=generator).tolist()
    
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]
    
    # Train subset
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    
    # Create a separate dataset for val with different transforms (no augmentation)
    val_dataset_full = datasets.GTSRB(
        root=data_dir,
        split='train',
        transform=get_transforms(is_train=False),
        download=True
    )
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

    # Test dataset (never augmented)
    test_dataset = datasets.GTSRB(
        root=data_dir,
        split='test',
        transform=get_transforms(is_train=False),
        download=True
    )
    
    # DataLoaders
    # num_workers=4 and pin_memory=True speed up data loading on machines with GPUs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def get_device():
    """Returns the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps') # For Apple Silicon
    else:
        return torch.device('cpu')
