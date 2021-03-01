import tensorflow as tf
from libs.layers.bigbird.bigbird_encoder import BigbirdEncoder


class PretrainingBigbirdEncoder(tf.keras.Model):
    def __init__(
        self, n_ch: int,
        hidden_size: int = 768,
        seq_size: int = 4096,
        intermediate_size: int = 3072,
        hidden_mask_prob: float = 0.0,  # Dropout probability for training
        num_attention_heads: int = 12,
        num_hidden_layers: int = 8,
        embedding_kernel: int = None,  # If None, use positional encoding instead of convolution.
        clip_embedding: int = None,  # Clip input value
    ):
        super().__init__()
        self.encoder = BigbirdEncoder(
            hidden_size=hidden_size,
            seq_size=seq_size,
            intermediate_size=intermediate_size,
            hidden_mask_prob=hidden_mask_prob,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            embedding_kernel=embedding_kernel,
            clip_embedding=clip_embedding,
        )
        self.model = tf.keras.Sequential([
            self.encoder,
            tf.keras.layers.Conv1D(64, kernel_size=21, padding='SAME'),
            tf.keras.layers.Conv1D(64, kernel_size=11, padding='SAME'),
            tf.keras.layers.Conv1D(n_ch, kernel_size=3, padding='SAME'),
            tf.keras.layers.Activation('tanh'),
        ])

    def call(self, xs, training=False, mask=None):
        if mask is None:
            mask = tf.ones_like(xs)
        return self.model(xs, mask=mask, training=training)

    def make_random_mask(self, xs):
        prob = tf.random.uniform((xs.shape[0], xs.shape[1], 1), 0.0, 1.0)
        drop_mask = prob <= 0.1
        random_mask = tf.math.logical_and(prob > 0.1, prob < 0.9)
        estimate_mask = prob > 0.1

        xs = tf.where(drop_mask, xs, 0.0)
        xs = tf.where(random_mask, xs, tf.random.uniform(tf.shape(xs), 0.0, 1.0))
        return xs, estimate_mask[..., 0]

    def train_step(self, data):
        xs = data
        with tf.GradientTape() as tape:
            xs_input, mask = self.make_random_mask(xs)
            ys_pred = self(xs_input, mask=mask, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.

            ys = tf.boolean_mask(xs, mask)
            ys_pred = tf.boolean_mask(ys_pred, mask)
            loss = self.compiled_loss(
                ys, ys_pred,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(ys, ys_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        xs = data
        xs_input, mask = self.make_random_mask(xs)
        # Compute predictions
        ys_pred = self(xs_input, mask=mask, training=False)
        # print(xs.shape, mask.shape, ys_pred.shape)
        # Updates the metrics tracking the loss
        ys = tf.boolean_mask(xs, mask)
        ys_pred = tf.boolean_mask(ys_pred, mask)
        self.compiled_loss(ys, ys_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(ys, ys_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
