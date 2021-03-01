import tensorflow as tf


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, name='InstanceNormalization'):
        super().__init__(name=name)
        self.epsilon = epsilon

    def call(self, x, training=False):
        mean, var = tf.nn.moments(
            x,
            axes=tf.range(1, tf.rank(x), dtype=tf.int32),
            keep_dims=True
        )
        return (x - mean) / tf.math.sqrt(var + self.epsilon)

    def get_config(self):
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config


class AdaIN(tf.keras.layers.Layer):
    def __init__(self, n_channels, epsilon=1e-5, name='AdaIN'):
        super().__init__(name=name)
        self.n_channels = n_channels
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma_fc = tf.keras.layers.Dense(
            self.n_channels, use_bias=True
        )
        self.beta_fc = tf.keras.layers.Dense(
            self.n_channels, use_bias=True
        )

    def call(self, x, training=False, style=None):
        x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_std = tf.sqrt(x_var + self.epsilon)

        x_norm = ((x - x_mean) / x_std)

        gamma = self.gamma_fc(style)
        beta = self.beta_fc(style)

        gamma = tf.reshape(gamma, shape=[-1, 1, 1, self.channels])
        beta = tf.reshape(beta, shape=[-1, 1, 1, self.channels])

        x = (1 + gamma) * x_norm + beta

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_channels': self.n_channels,
            'epsilon': self.epsilon,
        })
        return config
