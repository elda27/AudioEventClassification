import tensorflow as tf


class ConvRegressor(tf.keras.Model):
    def __init__(
        self, filters, kernel_size,
        conv_type=tf.keras.layers.Conv2D, activation=tf.nn.relu,
        **kwargs
    ):
        super().__init__()
        self.filters = filters
        self.conv_type = conv_type
        self.convs = []
        for n_filter in filters:
            self.convs.append(
                conv_type(n_filter, round_even(kernel_size), strides=2,
                          padding='SAME', activation=activation, **kwargs)
            )
            self.convs.append(conv_type(n_filter, kernel_size, padding='SAME',
                                        activation=activation, **kwargs))
        self.convs.append(tf.keras.layers.Conv2D(n_filter, kernel_size, padding='SAME'))

    @tf.function
    def call(self, x, training=False):
        y = x
        for conv in self.convs:
            y = conv(y, training=training)
        return y
