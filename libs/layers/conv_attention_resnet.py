from libs.layers.conv_attention import ConvAttention, AxialAttentionBlock

import tensorflow as tf
from itertools import zip_longest


class ResNetAxialAttentionBase(tf.keras.Model):
    """ResNet with Convolutional Bottleneck Attention Module"""

    def __init__(
        self, n_filters, n_freq_block, block_type,
        normalize_type=tf.keras.layers.BatchNormalization,
        kernel_size=3, block_kwargs={}
    ):
        super().__init__()

        self.n_blocks = len(n_filters)
        layers = [
            tf.keras.layers.Conv2D(
                64, kernel_size=7, strides=2,
                padding='same'
            ),
            normalize_type()
        ]
        resnet_layers = []
        for i, (n_filter, n_next_filter, n_freq) in enumerate(
                zip_longest(n_filters, n_filters[1:], n_freq_block)):
            layers.extend([
                block_type(n_filter, kernel_size=kernel_size, **block_kwargs)
                for j in range(n_freq)
            ])

            resnet_layers.append(
                tf.keras.Sequential(layers)
            )
            if n_next_filter is not None:
                layers = [
                    tf.keras.layers.Conv2D(
                        n_next_filter, kernel_size=kernel_size,
                        strides=kernel_size // 2 + (0 if kernel_size % 2 == 0 else 1),
                    )
                ]
        self.resnet_layers = tf.keras.Sequential(resnet_layers)

    # @tf.function
    def call(self, x, training=False):
        return self.resnet_layers(x, training=True)


class ResNet18(ResNetBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [2, 2, 2, 2], ResidualBlock,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )


class ResNet34(ResNetBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [3, 4, 6, 3], ResidualBlock,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )


class ResNet50(ResNetBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [3, 4, 6, 3], block_type=ResidualBottleneck,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )


class ResNet101(ResNetBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [3, 4, 23, 3], block_type=ResidualBottleneck,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )
