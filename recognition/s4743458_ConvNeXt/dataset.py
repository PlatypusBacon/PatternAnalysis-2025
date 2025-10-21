import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Callable
from fastcore.parallel import parallel
from pathlib import Path

class ADNIDataset(Dataset):
    def __init__(self, dir, split='train', transform=None):
        self.dir = os.path.join(dir, split)
        self.split = split
        self.transform = transform
        self.class_to_id = {'AD': 0, 'NC': 1}
        self.id_to_class = {0: 'AD', 1: 'NC'}
        self.images = []
        self._load()

    def _load(self):
        for class_name in ['AD', 'NC']:
            class_dir = os.path.join(self.dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                label = self.class_to_id[class_name]
                self.images.append((img_path, label))

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, id: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.images[id]
        image = Image.open(img_path).convert('L') #grayscale mri images
        image = self.transform(image)
        return image, label
    
def _check_size_worker(file):
    return Image.open(file).size

def check_sizes(dir, n_workers: int = 8) -> pd.Series:
    files = []
    for split in ['train', 'test']:
        for class_name in ['AD', 'NC']:
            class_dir = os.path.join(dir, split, class_name)
            for img_name in os.listdir(class_dir):
                files.append(os.path.join(class_dir, img_name))
    sizes = parallel(_check_size_worker, files, n_workers=8)
    size_counts = pd.Series(sizes).value_counts()
    print(size_counts)

def get_train_transforms(img_size: int = 224):
    """
    Training transforms with data augmentation.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
    ])


def get_test_transforms(img_size: int = 224):
    """
    Test/validation transforms without augmentation.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
    ])

def get_class_counts() -> torch.Tensor:
    """
    get count of true classes to determine dataset imbalance
    """
    datasetTrain = ADNIDataset('ADNI/AD_NC')
    datasetTest = ADNIDataset('ADNI/AD_NC', 'test')
    labels1 = [label for _, label in datasetTrain.images]
    class_counts1 = np.bincount(labels1)
    labels2 = [label for _, label in datasetTest.images]
    class_counts2 = np.bincount(labels2)
    print(class_counts1)
    print(class_counts2)

def make_dataloaders(
    dir, 
    batch_size: int = 32, 
    img_size: int = 224, 
    num_workers: int = 4,
    pin_memory: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.
    
    Args:
        dir: Root directory (e.g., 'AD_NC')
        batch_size: Batch size for training and testing
        img_size: Target image size (224, 256, or 384)
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader, test_loader
    """
    
    # Get transforms
    train_transform = get_train_transforms(img_size)
    test_transform = get_test_transforms(img_size)
    
    # Create datasets
    train_dataset = ADNIDataset(dir, split='train', transform=train_transform)
    test_dataset = ADNIDataset(dir, split='test', transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}")
    
    return train_loader, test_loader
