import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Optional
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
    """
    Layer Normalisation
    computes mean and variance for features
    normalises input, and applies scaling
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)

        
class DropPath(nn.Module):
    def __init__(self, prob: float = 0.0):
        super().__init__()
        self.prob = prob

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
    def __init__(self, dim: int, 
                 drop_path: float = 0.0, 
                 layer_scale_init: float = 1e-6,
                 mlp_ratio: float = 4.0):
        super().__init__()
        # Depthwise Conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # groups = dim for depthwise
        # 
        self.norm = LayerNorm2d(dim)


        # conv2d


class ConvNeXt(nn.Module):
    """
    ConvNeXt model
    """
    def __init__(self, in_chans: int = 1,
                 num_classes: int = 2,
                 depths: List[int] = [3,3,9,3],
                 dims: List[int] = [96,192,384,768],
                 drop_path_prob: float = 0.0,
                 layer_scale_init: float = 1e-6,
                 head_init_scale: float = 1.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_prob, sum(depths))]
        # Layer Iteration
        cur = 0 
        for i in range(4):
            # Downsampling layer
            if i > 0:
                downsample = nn.Sequential(
                    LayerNorm2d(dims[i-1]),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2)
                )
            else: # first layer no downsample
                downsample = nn.Identity()
            
            stage = nn.Sequential(
                downsample,
                *[ConvNeXtBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

