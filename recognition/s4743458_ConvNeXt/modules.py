import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")
from PIL import Image
"""
Input: []

ConvNeXt notes:
4 stages:
Stage 1: resolution (64,60)
Stage 2: resolution (32,30)
Stage 3: resolution (16,15)
Stage 4: resolution (8,7-8)?
"""

class LayerNorm2d(nn.Module):
    def __init__(self, norm_shape: int):
        super().__init__()

class DropPath(nn.Module):
    def __init__(self):
        pass

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block:
    Depthwise Conv2d -> ->
    Layer Norm ->
    Conv2d ->
    GELU ->
    Conv2d ->
    Layer Scale ->
    Drop Path -> o <-
    """
    def __init__(self, in_channels):
        pass


class ConvNeXt(nn.Module):
    def __init__(self):
        pass