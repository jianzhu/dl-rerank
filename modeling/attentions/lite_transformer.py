import tensorflow as tf
from modeling.attentions.transformer import MultiHeadAttention
from modeling.attentions.transformer import FeedForwardNetwork
from modeling.attentions.light_conv import LightConvBlock


class LSRAAttentionBlock(tf.keras.layers.Layer):
    """LSRA attention block

       ref: Lite Transformer with Long-Short Range Attention
       arxiv: https://arxiv.org/abs/2004.11886
    """

    def __init__(self, head_num, hidden_size, filter_size, kernel_size, dropout_rate):
        super(LSRAAttentionBlock, self).__init__()

        self.multi_head_attention = MultiHeadAttention(head_num=head_num,
                                                       hidden_size=hidden_size // 2,
                                                       dropout_rate=dropout_rate)
        self.ffn = FeedForwardNetwork(filter_size=filter_size,
                                      hidden_size=hidden_size, dropout_rate=dropout_rate)

        self.light_conv = LightConvBlock(dropout_rate=dropout_rate,
                                         kernel_size=kernel_size)

    def call(self, inputs, training=False):
        # [B, T, H]
        from_tensor = inputs[0]
        # [B]
        sequence_len = inputs[1]

        left, right = tf.split(from_tensor, 2, axis=-1)

        # do global attention with transformer
        left = self.multi_head_attention([left, sequence_len], training=training)
        # do local attention with light convolution
        right = self.light_conv(right, training=training)
        output_tensor = tf.concat([left, right], axis=-1)
        output_tensor = self.ffn(output_tensor, training=training)
        return output_tensor


class LiteTransformer(tf.keras.layers.Layer):
    """Lite Transformer with Long-Short Range Attention"""

    def __init__(self, layer_num,
                 head_num, hidden_size, filter_size, kernel_size, dropout_rate):
        super(LiteTransformer, self).__init__()
        self.attention_blocks = [LSRAAttentionBlock(head_num=head_num,
                                                    hidden_size=hidden_size,
                                                    filter_size=filter_size,
                                                    kernel_size=kernel_size,
                                                    dropout_rate=dropout_rate) for _ in range(layer_num)]

    def call(self, inputs, training=False):
        """inputs contains following two tensor

           input:
               input tensor: shape (B, T, H)
               input sequence length tensor: shape (B,)
           output:
               output tensor: shape (B, T, H)
        """
        output_tensor = inputs[0]
        sequence_len = inputs[1]
        for attention_block in self.attention_blocks:
            output_tensor = attention_block([output_tensor, sequence_len], training=training)
        return output_tensor
