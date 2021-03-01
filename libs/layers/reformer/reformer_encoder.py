import tensorflow as tf
from libs.layers.reformer.TFreformers import TFReformer
from libs.layers.attention_utils import positional_encoding


class ReformerEncoder(tf.keras.Model):
    def __init__(
        self,
        hidden_size: int = 768,
        seq_size: int = 4096,
        intermediate_size: int = 3072,
        hidden_mask_prob: float = 0.0,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 8,
        embedding_kernel: int = None,  # If None, use positional encoding instead of convolution.
    ):
        super().__init__()

        embedding_layers = []
        if hidden_mask_prob > 0:
            embedding_layers.append(tf.keras.layers.Dropout(hidden_mask_prob))
        if embedding_kernel is not None:
            embedding_layers.append(
                tf.keras.layers.Conv1D(hidden_size, kernel_size=embedding_kernel)
            )

        self.embedding = tf.keras.Sequential(embedding_layers)
        self.positional_encoding = positional_encoding(seq_size, hidden_size)

        self.encoder = TFReformer()

    def call(self, xs, mask=None, training=False):
        assert mask is not None
        embedding_output = self.embedding(xs, training=training)
        sequence_output = self.encoder(embedding_output, mask, training=training)
        return sequence_output
