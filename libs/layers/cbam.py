# Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and So Kweon, "CBAM: Convolutional Block Attention Module", in ECCV2018
# Reimplementation of original source code by tensorflow 2.0
# Some arguments is changed but network architecture is the same.
# If you
import tensorflow as tf


class CBR(tf.keras.Model):
    def __init__(
        self, n_filters, kernel_size, activation=tf.nn.relu,
        **kwargs
    ):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            n_filters, kernel_size=kernel_size,
            **kwargs
        )
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=0.01, epsilon=1e-5,
        )
        self.activation = activation

    # @tf.function
    def call(self, x, training=False):
        x = self.conv(x, training=training)
        x = self.bn(x)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x


class ChannelAttentionLayer(tf.keras.Model):
    def __init__(
        self, n_channel, activation=tf.nn.sigmoid,
        reduction_ratio=8, mlp_activation=tf.nn.relu,
        poolings=[
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.GlobalMaxPool2D()
        ],
    ):
        super().__init__()
        self.activation = activation
        self.poolings = poolings
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                n_channel // reduction_ratio,
                activation=mlp_activation
            ),
            tf.keras.layers.Dense(
                n_channel,
                activation=None
            )
        ])

    # @tf.function
    def call(self, x, training=False):
        ys = []
        for pool in self.poolings:
            ys.append(
                self.mlp(pool(x), training=training)
            )
        return tf.broadcast_to(
            tf.expand_dims(tf.expand_dims(
                self.activation(sum(ys)),
                axis=1
            ), axis=2),
            tf.shape(x)
        )


class ChannelPool(tf.keras.Model):
    def __init__(self):
        super().__init__()

    # @tf.function
    def call(self, x, training=False):
        return tf.concat([
            tf.expand_dims(tf.reduce_max(x, axis=-1), axis=-1),
            tf.expand_dims(tf.reduce_mean(x, axis=-1), axis=-1)
        ], axis=-1)


class SpatialAttentionLayer(tf.keras.Model):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.spatial = CBR(
            1, kernel_size=kernel_size, strides=1,
            padding='same',
            activation=None
        )
        self.compress = ChannelPool()

    # @tf.function
    def call(self, x, training=False):
        y = self.compress(x, training=training)
        y = self.spatial(y, training=training)
        scale = tf.nn.sigmoid(y)
        return x * scale


class CBAM(tf.keras.Model):
    def __init__(
        self, gate_channels, reduction_ratio=16,
        poolings=[
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.GlobalMaxPool2D()
        ],
        no_spatial=False
    ):
        super().__init__()
        self.channel_gate = ChannelAttentionLayer(
            gate_channels, reduction_ratio=reduction_ratio,
            poolings=poolings
        )
        self.no_spatial = no_spatial
        if not no_spatial:
            self.spatial_gate = SpatialAttentionLayer()

    # @tf.function
    def call(self, x, training=False):
        y = self.channel_gate(x, training=training)
        if not self.no_spatial:
            y = self.spatial_gate(y, training=training)
        return y
