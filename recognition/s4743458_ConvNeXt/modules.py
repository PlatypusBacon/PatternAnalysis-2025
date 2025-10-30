import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU.")


class DropPath(nn.Module):
    """Drops entire residual branch with given probability."""
    def __init__(self, prob: float = 0.0):
        super().__init__()
        self.prob = prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.prob > 0:
            keep_prob = 1 - self.prob
            mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
            x = x.div(keep_prob) * mask
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for channels-first tensors."""
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma[:, None, None] * x
        x = self.drop_path(x) + residual
        return x


class ConvNeXt(nn.Module):
    """Simplified ConvNeXt with channels-first normalization."""
    def __init__(
        self,
        in_chans: int = 1,
        num_classes: int = 2,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_prob: float = 0.0,
        layer_scale_init: float = 1e-6,
        head_init_scale: float = 1.0,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0]),
        )
        self.downsample_layers.append(self.stem)

        # Downsample stages
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                nn.BatchNorm2d(dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        # Stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_prob, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init=layer_scale_init,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.BatchNorm2d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        self.dropout = nn.Dropout(0.5)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.downsample_layers[0](x)
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.downsample_layers[i + 1](x)
        x = self.norm(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.dropout(x)
        x = self.head(x)
        return x


def convnext_small(drop_path_rate: float = 0.2, layer_scale: float = 1e-6):
    return ConvNeXt(
        in_chans=1,
        depths=[3,3, 7, 3],
        dims=[64, 128, 256, 512],
        num_classes=2,
        drop_path_prob=drop_path_rate,
        layer_scale_init=layer_scale,
    ).to(device)

def convnext_small_2(drop_path_rate: float = 0.2, layer_scale: float = 1e-6):
    return ConvNeXt(
        in_chans=1,
        depths=[3,3, 9, 3],
        dims=[72, 144, 288, 576],
        num_classes=2,
        drop_path_prob=drop_path_rate,
        layer_scale_init=layer_scale,
    ).to(device)


def convnext_medium(drop_path_rate: float = 0.2, layer_scale: float = 1e-6):
    return ConvNeXt(
        in_chans=1,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        num_classes=2,
        drop_path_prob=drop_path_rate,
        layer_scale_init=layer_scale,
    ).to(device)


def convnext_2(drop_path_rate: float = 0.2, layer_scale: float = 1e-6):
    return ConvNeXt(
        in_chans=1,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        num_classes=2,
        drop_path_prob=drop_path_rate,
        layer_scale_init=layer_scale,
    ).to(device)


def convnext_3(drop_path_rate: float = 0.2, layer_scale: float = 1e-6):
    return ConvNeXt(
        in_chans=1,
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        num_classes=2,
        drop_path_prob=drop_path_rate,
        layer_scale_init=layer_scale,
    ).to(device)
