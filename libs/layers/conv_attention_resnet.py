from libs.layers.conv_attention import ConvAttention, AxialAttentionBlock
from libs.layers.resnet import ResNetBase

import tensorflow as tf
from itertools import zip_longest


class AttentionResNet18(ResNetBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [2, 2, 2, 2], ConvAttention,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )


class AttentionResNet34(ResNetBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [3, 4, 6, 3], ConvAttention,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )


class AttentionResNet50(ResNetBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [3, 4, 6, 3], ConvAttention,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )


class AttentionResNet101(ResNetBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [3, 4, 23, 3], ConvAttention,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )


class AxialAttentionResNet18(ResNetBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [2, 2, 2, 2], AxialAttentionBlock,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )


class AxialAttentionResNet34(ResNetBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [3, 4, 6, 3], AxialAttentionBlock,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )


class AxialAttentionResNet50(ResNetBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [3, 4, 6, 3], AxialAttentionBlock,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )


class AxialAttentionResNet101(ResNetBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [3, 4, 23, 3], AxialAttentionBlock,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )
