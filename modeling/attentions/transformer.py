import tensorflow as tf

from absl import flags

FLAGS = flags.FLAGS


class FeedForwardNetwork(tf.keras.layers.Layer):
    """Position-wise Feed-Forward Networks"""

    def __init__(self, filter_size, hidden_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.filter_layer = tf.keras.layers.Dense(
            units=filter_size, activation='relu', name="filter_layer")
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        self.output_layer = tf.keras.layers.Dense(
            units=hidden_size, name="output_layer")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-7)

    def call(self, inputs, training=False):
        """Return outputs of the feedforward network.

        Args:
          inputs: tensor with shape (batch_size, length, hidden_size)
          training: boolean, whether in training mode or not.

        Returns:
          Output of the feedforward network.
          tensor with shape (batch_size, length, hidden_size)
        """
        output = self.filter_layer(inputs)
        output = self.dropout_layer(output, training=training)
        output = self.output_layer(output)
        output = self.layer_norm(inputs + output)
        return output


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-Head Attention"""

    def __init__(self, head_num, hidden_size, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % head_num != 0:
            raise ValueError(
                "hidden_size (%d) is not a multiple of head_num (%d)" % (hidden_size, head_num))
        self.head_num = head_num
        self.hidden_size = hidden_size
        self.k_size = hidden_size // head_num
        self.v_size = hidden_size // head_num

        self.query_dense = tf.keras.layers.Dense(units=hidden_size, use_bias=False)
        self.key_dense = tf.keras.layers.Dense(units=hidden_size, use_bias=False)
        self.value_dense = tf.keras.layers.Dense(units=hidden_size, use_bias=False)
        self.output_dense = tf.keras.layers.Dense(units=hidden_size, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        def reshape_transpose(x, batch_size, seq_len, head_num, hidden_size):
            # [B, F, N, K]
            x = tf.reshape(x, [batch_size, seq_len, head_num, hidden_size])
            # [B, N, F, k]
            return tf.transpose(x, [0, 2, 1, 3])

        # [B, F, H]
        from_tensor = inputs[0]
        to_tensor = inputs[0]
        # [B]
        sequence_len = inputs[1]

        # [B, F, H]
        query = self.query_dense(from_tensor)
        key = self.key_dense(to_tensor)
        value = self.value_dense(to_tensor)

        batch_size = tf.shape(from_tensor)[0]
        # [B, N, F, K]
        query = reshape_transpose(query, batch_size, -1, self.head_num, self.k_size)
        # [B, N, F, K]
        key = reshape_transpose(key, batch_size, -1, self.head_num, self.k_size)
        # [B, N, F, V]
        value = reshape_transpose(value, batch_size, -1, self.head_num, self.v_size)

        # [B, N, F, F]
        attention_scores = tf.matmul(query, key, transpose_b=True)
        normalizer = tf.cast(1.0 / tf.sqrt(float(self.k_size)), dtype=attention_scores.dtype)
        # [B, N, F, F]
        attention_scores = tf.multiply(attention_scores, normalizer)
        # [B, F]
        sequence_mask = tf.sequence_mask(sequence_len)
        # [B, 1, 1, F]
        sequence_mask = tf.expand_dims(tf.expand_dims(sequence_mask, axis=1), axis=1)
        # [B, 1, 1, F]
        adder = tf.cast((1.0 - tf.cast(sequence_mask, tf.float32)) * -10000.0, dtype=attention_scores.dtype)
        # [B, N, F, F]
        attention_weight = tf.nn.softmax(attention_scores + adder)
        attention_weight = self.dropout(attention_weight, training=training)
        # [B, N, F, V] = [B, N, F, F] x [B, N, F, V]
        output = tf.matmul(attention_weight, value)
        # [B, F, N, V]
        output = tf.transpose(output, [0, 2, 1, 3])
        # [B, F, H]
        output = tf.reshape(output, [batch_size, -1, self.hidden_size])
        output = self.layer_norm(output + from_tensor)
        return output


class AttentionBlock(tf.keras.layers.Layer):
    """Transformer attention block"""

    def __init__(self, head_num, hidden_size, filter_size, dropout_rate):
        super(AttentionBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(head_num=head_num,
                                                       hidden_size=hidden_size,
                                                       dropout_rate=dropout_rate)
        self.ffn = FeedForwardNetwork(filter_size=filter_size,
                                      hidden_size=hidden_size, dropout_rate=dropout_rate)

    def call(self, inputs, training=False):
        output_tensor = self.multi_head_attention(inputs, training=training)
        output_tensor = self.ffn(output_tensor, training=training)
        return output_tensor


class Transformer(tf.keras.layers.Layer):
    """Transformer encoder model

       Ref: Attention Is All You Need
       arxiv: https://arxiv.org/abs/1706.03762
    """

    def __init__(self, layer_num,
                 head_num, hidden_size, filter_size, dropout_rate):
        super(Transformer, self).__init__()
        self.attention_blocks = [AttentionBlock(head_num=head_num,
                                                hidden_size=hidden_size,
                                                filter_size=filter_size,
                                                dropout_rate=dropout_rate) for _ in range(layer_num)]

    def call(self, inputs, training=False):
        """inputs contains following two tensor

           input tensor: shape (B, T, H)
           input sequence length tensor: shape (B,)
        """
        output_tensor = inputs[0]
        sequence_len = inputs[1]
        for attention_block in self.attention_blocks:
            output_tensor = attention_block([output_tensor, sequence_len], training=training)
        return output_tensor
