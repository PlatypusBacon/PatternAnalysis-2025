"""
Input: []

ConvNeXt notes:
4 stages:

"""

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
    def __init(self, )