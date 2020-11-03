import tensorflow as tf


class ConvAttention(tf.keras.Model):
    def __init__(self, n_filters, n_heads=8, bias=False, with_skip_connection=True):
        assert (n_filters % n_heads) == 0
        super().__init__()
        self.conv_qkv = tf.keras.layers.Conv2D(
            n_filters * 2, kernel_size=1, stride=1, padding=1, bias=bias,
            padding='same'
        )
        self.with_skip_connection = with_skip_connection

    def call(self, x):
        qkv = self.conv_qkv(x)  # [N, H, W, n_filters*2]
        qk, v = tf.split(qkv, 2, axis=-1)  # [N, H, W, n_filters]
        q, k = tf.split(qk, 2, axis=-1)  # [N, H, W, n_filters//2]
        w = tf.nn.softmax(tf.matmul(q, k, transpose_b=True), axis=-1)
        y = tf.nn.matmul(w, v)
        if with_skip_connection:
            y = x + y
        return y, w


class AxialAttention(tf.keras.Model):
    def __init__(
        self, n_filters, n_heads=8, embed_size=56,
        bias=False, width_attention=False,
        normalize_layer=tf.keras.layers.BatchNormalization
    ):
        assert n_filters % n_heads == 0
        super().__init__()
        self.width_attention = width_attention
        self.n_heads = n_heads
        self.n_channel = n_filters // n_heads
        self.embed_size = embed_size

        # Multi-head self attention
        self.conv_qkv = tf.keras.layers.Conv2D(
            n_filters * 2, kernel_size=1, stride=1,
            padding=0, bias=False, padding='same'
        )
        self.norm_qkv = normalize_layer()
        self.norm_attention = normalize_layer()
        self.norm_output = normalize_layer()

        # Position embedding
        self.relative_embedding = self.add_variable(
            "relative_embedding",
            tf.random.normal((
                self.n_channel * 2, embed_size * 2 - 1
            )),
        )
        query_index = tf.expand_dims(tf.range(embed_size), axis=0)
        key_index = tf.expand_dims(tf.range(embed_size), axis=1)
        self.relative_index = tf.reshape(
            tf.key_index - query_index + embed_size - 1,
            (-1,)
        )

    def call(self, x, training=False):
        if self.width:
            x = tf.transpose(x, (0, 1, 3, 2))  # [N, H, W, C] -> [N, H, C, W]
        else:
            x = tf.transpose(x, (0, 2, 3, 1))  # [N, H, W, C] -> [N, W, C, H]

        N, A, C, F = tf.shape(x,)

        # Transformations
        qkv = self.bn_qkv(self.conv_qkv(x))
        q, k, v = tf.split(
            qkv, [self.n_filters, self.n_filters, self.n_filters],
            axis=1
        )

        qk, v = tf.split(qkv, 2, axis=-1)  # [N, H, W, n_filters]
        q, k = tf.split(qk, 2, axis=-1)  # [N, H, W, n_filters//2]
        # Calculate position embedding
        all_embeddings = tf.reshape(
            tf.gather(self.relative, self.relative_index),
            (self.embed_size * 2, self.embed_size, self.embed_size)
        )
        q_embedding, k_embedding, v_embedding = tf.split(
            all_embeddings, [self.n_filters, self.n_filters, self.n_filters * 2], axis=1
        )
        qr = tf.transpose(tf.matmul(q, q_embedding), (0, 2, 1))
        kr = tf.matmul(k, k_embedding)
        qk = tf.matmul(q, k)
        stacked_similarity = tf.concat([qk, qr, kr], axis=1)
        stacked_similarity = tf.reduce_sum(
            tf.reshape(
                self.bn_similarity(stacked_similarity),
                (N * A, 3, self.n_channel, F, F)
            )
        )
        similarity = tf.nn.softmax(stacked_similarity, axis=-1)
        sv = tf.matmul(similarity, v)
        sve = tf.matmul(similarity, v_embedding)
        y = self.bn_output(sv + sve, training=training)
        if self.width:
            y = tf.transpose(y, (0, 1, 3, 2))  # [N, H, C, W] -> [N, H, W, C]
        else:
            y = tf.transpose(y, (0, 3, 1, 2))  # [N, W, C, H] -> [N, H, W, C]

        return y


class AxialAttentionBlock(tf.keras.Model):
    def __init__(
        self, n_filters, n_heads=8, embed_size=56,
        bias=False, width_attention=False,
        normalize_layer=tf.keras.layers.BatchNormalization
    ):
        super().__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = tf.keras.layers.Conv2D(
            n_filters, kernel_size=1
        )
        self.norm1 = normalize_layer()
        self.hight_block = AxialAttention(
            n_filters, n_heads=n_heads, embed_size=embed_size,
            bias=bias, width_attention=width_attention,
            normalize_layer=normalize_layer, width=False
        )
        self.width_block = AxialAttention(
            n_filters, n_heads=n_heads, embed_size=embed_size,
            bias=bias, width_attention=width_attention,
            normalize_layer=normalize_layer, width=True
        )
        self.conv_up = tf.keras.layers.Conv2D(
            n_filters * 2, kernel_size=1, padding='same',
        )
        self.norm2 = normalize_layer()

    def forward(self, x, training=False):

        y = self.conv_down(x)
        y = self.norm1(y, training=training)
        y = tf.nn.relu(y)

        y = self.hight_block(y, training=training)
        y = self.width_block(y, training=training)
        y = tf.nn.relu(y)

        y = self.conv_up(y, training=training)
        y = self.norm2(y, training=training)

        return tf.nn.relu(y + x)
