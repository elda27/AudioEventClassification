import tensorflow as tf
from libs.layers.normalize import InstanceNormalization


class CoordinateConv2D(tf.keras.Model):
    def __init__(self, n_filters, kernel_size, with_r=True, with_boundary=False, **kwargs):
        """Coordinate convolution is an exntesion to convolution operation for 2d image
        with index coordinates.

        Args:
            n_filters ([type]): n filters output
            kernel_size ([type]): kernel_size
            with_r (bool, optional): If True, with squared root index coordinates. Defaults to True.
            with_boundary (bool, optional): . Defaults to False.
        """
        super().__init__()
        self.with_r = with_r
        self.with_boundary = with_boundary

        self.conv = tf.keras.layers.Conv2D(
            n_filters, kernel_size,
            **kwargs
        )

    def call(self, x, training=False, boundary=None):
        """ Forward network

        Args:
            x (tf.Tensor): input tensor
            training (bool, optional): training flag. Defaults to False.
            boundary (tf.Tensor, optional): Boundary information.
                Defaults to None.

        Returns:
            tf.Tensor: regression result
        """
        batch_size, x_dim, y_dim = tf.shape(x)[:2]

        xx_indices = tf.tile(
            tf.reshape(tf.range(x_dim), (1, 1, 1, x_dim)),
            (1, batch_size, y_dim, 1)
        )

        yy_indices = tf.tile(
            tf.reshape(tf.range(y_dim), (1, y_dim, 1, 1))
            (batch_size, 1, x_dim, 1)
        )

        xx_indices = tf.divide(xx_indices, x_dim - 1)
        yy_indices = tf.divide(yy_indices, y_dim - 1)

        xx_indices = tf.cast(
            tf.subtract(tf.multiply(xx_indices, 2.0), 1.0),
            dtype=x.dtype
        )
        yy_indices = tf.cast(
            tf.subtract(tf.multiply(yy_indices, 2.0), 1.0),
            dtype=x.dtype
        )

        xs = tf.concat([x, xx_indices, yy_indices], axis=-1)

        if self.with_r:
            rr = tf.sqrt(
                (xx_indices - 0.5) ** 2 + (yy_indices - 0.5) ** 2
            )
            xs = tf.concat([xs, rr], axis=-1)

        if self.with_boundary and (boundary is not None):
            boundary_indices = tf.clip_by_value(boundary, 0.0, 1.0)
            xx_boundary_indices = tf.where(
                boundary_indices > 0.05, xx_indices, 0)
            yy_boundary_indices = tf.where(
                boundary_indices > 0.05, yy_indices, 0)

            xs = tf.concat(
                [xs, xx_boundary_indices, yy_boundary_indices], axis=-1)

        return self.conv(xs, training=training)


class ConvBlockForHourGlass(tf.keras.Model):
    """Convolution layer for hourglass module (original paper version)
    """

    def __init__(
        self, n_filters: int,
        norm_type: str = 'pre', conv_layer=tf.keras.layers.Conv2D,
        norm_layer: tf.keras.layers.Layer = InstanceNormalization,
        **kwargs
    ):
        """In this implementation, the batch normalization is replaced by instance normalization.
        If you want to use original implementation, you should set the norm_layer
        to `tf.keras.layers.BatchNormalization`.

        Args:
            n_filters (int): number of filters
            norm_type (str, optional): position of normalization layer. Defaults to 'pre'.
            conv_layer (tf.keras.layers.Layer, optional): Convolution type. Defaults to tf.keras.layers.Conv2D.
            norm_layer (tf.keras.layers.Layer, optional): Normalization type. Defaults to InstanceNormalization.
        """
        super().__init__()
        blocks = []

        def create_block(n_filters, kernel_size):
            layers = []
            if norm_type == 'pre':
                layers.append(norm_layer())

            layers.append(conv_layer(
                n_filters, kernel_size, padding='SAME', **kwargs))

            if norm_type == 'post':
                layers.append(norm_layer())
            return tf.keras.Sequential(layers)

        blocks.append(create_block(n_filters // 2, 1))
        blocks.append(create_block(n_filters // 2, 3))
        blocks.append(create_block(n_filters, 1))

    def call(self, x, training=False):
        y = x
        for block in self.blocks:
            y = block(y, training=training)
        return y + x


class ConvBlockForFAN(tf.keras.Model):
    """Improved hourglass architecture proposed by Adrain Bulat.
    This implementation is used on the FAN (Face Alignment Network).
    See detail: https://arxiv.org/abs/1703.07332
    """

    def __init__(
        self, n_filters: int,
        norm_type: str = 'pre', conv_layer=tf.keras.layers.Conv2D,
        norm_layer: tf.keras.layers.Layer = InstanceNormalization,
        **kwargs
    ):
        """In this implementation, the batch normalization is replaced by instance normalization.
        If you want to use original implementation, you should set the norm_layer
        to `tf.keras.layers.BatchNormalization`.

        Args:
            n_filters (int): number of filters
            norm_type (str, optional): position of normalization layer. Defaults to 'pre'.
            conv_layer (tf.keras.layers.Layer, optional): Convolution type. Defaults to tf.keras.layers.Conv2D.
            norm_layer (tf.keras.layers.Layer, optional): Normalization type. Defaults to InstanceNormalization.
        """
        assert (n_filters % 2) != 9,\
            f"n_filters shoukd be even value. Actual:{n_filters}"
        super().__init__()
        blocks = []

        def create_block(n_filters, kernel_size):
            layers = []
            if norm_type == 'pre':
                layers.append(norm_layer())

            layers.append(conv_layer(
                n_filters, kernel_size, padding='SAME', **kwargs))

            if norm_type == 'post':
                layers.append(norm_layer())
            return tf.keras.Sequential(layers)

        blocks.append(create_block(n_filters // 2, 3))
        blocks.append(create_block(n_filters // 4, 3))
        blocks.append(create_block(n_filters // 4, 3))

    def call(self, x, training=False):
        y = x
        ys = []
        for block in self.blocks:
            y = block(y, training=training)
            ys.append(y)
        return x + tf.concat(ys, axis=-1)


class HourGlass(tf.keras.Model):
    def __init__(self, n_depth: int, n_filters: int = 256, conv_block=ConvBlockForFAN, **kwargs):
        """HourGlass module
        See detail: https://arxiv.org/abs/1603.06937

        Args:
            n_depth (int): number of repeat the encoder/decoder.
            n_filters (int): number of filters of the convolution
        """

        super().__init__()
        self.conv_before_downsample = [
            (conv_block(n_filters), tf.keras.layers.AveragePooling2D())
            for i in range(n_depth)
        ]
        self.conv_before_downsample = [
            (conv_block(n_filters), tf.keras.layers.UpSampling2D())
            for i in range(n_depth)
        ]
        self.conv_latent = conv_block(n_depth)

    def call(self, x, training=False, mask=None):
        y = x
        data = []
        for (conv, pool) in self.conv_before_downsample:
            y = conv(y, training=training)
            y = pool(y)
            data.append(y)

        y = self.conv_latent(y, training=training)

        for (conv, resampler), d in zip(self.conv_before_downsample, reversed(data)):
            y = conv(y, training=training)
            y = resampler(y)
            y = y + d
        return y


class HeatmapRegressor(tf.keras.Model):
    """Heatmap regressor for Face Alignment (FaceAlignmentNetwork).
    See detail: https://arxiv.org/abs/1703.07332
    """

    def __init__(self, n_depth=1, n_landmarks=98, n_units=256, norm_type=InstanceNormalization):
        super().__init__()
        self.base_net = tf.keras.Sequential([
            CoordinateConv2D(
                n_units, kernel_size=7, strides=2, padding='VALID'
            ),
            norm_type(),
            tf.keras.layers.Relu(),
            ConvBlockForFAN(128),
            tf.keras.layers.AveragePooling2D(),
            ConvBlockForFAN(128),
            ConvBlockForFAN(256),
        ])

    def call(self, x, training=False):
        y = self.base_net(x, training=training)
