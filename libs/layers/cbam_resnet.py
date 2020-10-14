import tensorflow as tf
from libs.layers import cbam
from itertools import zip_longest


class ResidualCBAM(tf.keras.Model):
    """For ResNet 18/34"""

    def __init__(self, n_filters, kernel_size, cbam_kwargs={}):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            n_filters, kernel_size,
            padding='same',
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.activation = tf.nn.relu

        self.conv2 = tf.keras.layers.Conv2D(
            n_filters, kernel_size,
            padding='same',
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        _cbam_kwargs = dict(gate_channels=n_filters)
        _cbam_kwargs.update(cbam_kwargs)
        self.cbam = cbam.CBAM(_cbam_kwargs.pop('gate_channels'), **cbam_kwargs)

    # @tf.function
    def call(self, x, training=False):
        y = x
        y = self.activation(self.bn1(self.conv1(y, training=training), training=training))
        y = self.bn2(self.conv2(y, training=training), training=training)
        y = self.cbam(y, training=training)
        return self.activation(y + x)


class ResidualBottleneckCBAM(tf.keras.Model):
    """For ResNet 50/101/152"""

    def __init__(
        self, n_filters, kernel_size,
        internal_reduction=4, cbam_kwargs={}
    ):
        super().__init__()
        self.activation = tf.nn.relu

        self.conv0 = tf.keras.layers.Conv2D(
            n_filters // internal_reduction, 1,
            padding='same'
        )
        self.bn0 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2D(
            n_filters, kernel_size,
            padding='same',
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            n_filters, kernel_size,
            padding='same',
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        _cbam_kwargs = dict(gate_channels=n_filters)
        _cbam_kwargs.update(cbam_kwargs)
        self.cbam = cbam.CBAM(
            _cbam_kwargs.pop('gate_channels'),
            **_cbam_kwargs
        )

    # @tf.function
    def call(self, x, training=False):
        y = x
        y = self.activation(self.bn0(self.conv0(y, training=training), training=training))
        y = self.activation(self.bn1(self.conv1(y, training=training), training=training))
        y = self.bn2(self.conv2(y, training=training), training=training)
        y = self.cbam(y, training=training)
        return self.activation(y + x)


class ResNetCBAMBase(tf.keras.Model):
    """ResNet with Convolutional Bottleneck Attention Module"""

    def __init__(self, n_filters, n_freq_block, block_type, kernel_size=3, block_kwargs={}):
        super().__init__()

        self.n_blocks = len(n_filters)
        layers = [
            tf.keras.layers.Conv2D(
                64, kernel_size=7, strides=2,
                padding='same'
            ),
            tf.keras.layers.BatchNormalization()
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


class ResNetCBAM18(ResNetCBAMBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [2, 2, 2, 2], ResidualCBAM,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )


class ResNetCBAM34(ResNetCBAMBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [3, 4, 6, 3], ResidualCBAM,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )


class ResNetCBAM50(ResNetCBAMBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [3, 4, 6, 3], block_type=ResidualBottleneckCBAM,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )


class ResNetCBAM101(ResNetCBAMBase):
    def __init__(self, kernel_size=3, block_kwargs={}):
        super().__init__(
            [64, 128, 256, 512], [3, 4, 23, 3], block_type=ResidualBottleneckCBAM,
            kernel_size=kernel_size, block_kwargs=block_kwargs
        )
