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
    def __init__(self, dir, split='train'):
        self.dir = os.path.join(dir, split)
        self.split = split
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
def make_dataloaders(dir, batch_size: int = 32, img_size: int = 224, num_workers: int = 4):
    pass