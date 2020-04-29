import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    """Convolutional Sequence to Sequence Learning

       arxiv: https://arxiv.org/abs/1705.03122
    """

    def __init__(self):
        super(GLU, self).__init__()

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(units=input_shape[-1]*2)

    def call(self, inputs, training=False):
        # (B, T, 2H)
        x = self.dense(inputs)
        a, b = tf.split(x, 2, axis=-1)
        # (B, T, H)
        return a * tf.sigmoid(b)


class LightConvBlock(tf.keras.layers.Layer):
    """Pay less attention with lightweight and dynamic convolutions

       arxiv: https://arxiv.org/pdf/1901.10430.pdf
    """

    def __init__(self, dropout_rate, kernel_size):
        super(LightConvBlock, self).__init__()

        self.kernel_size = kernel_size
        self.glu = GLU()
        self.drop = tf.keras.layers.Dropout(rate=dropout_rate)

    def build(self, input_shape):
        shape = (self.kernel_size, 1, input_shape[-1], 1)
        self.filter = self.add_weight(name='filter', shape=shape,
                                      initializer='random_normal',
                                      trainable=True)
        self.bias = self.add_weight(name='bias', shape=input_shape[-1],
                                    initializer='zeros',
                                    trainable=True)
        self.dense = tf.keras.layers.Dense(units=input_shape[-1])

    def call(self, inputs, training=False):
        # (B, T, C)
        x = self.glu(inputs)
        # (B, T, 1, C)
        x = tf.expand_dims(x, axis=-2)

        filter = self.drop(tf.nn.softmax(self.filter, axis=0), training=training)
        x = tf.nn.depthwise_conv2d(input=x,
                                   filter=filter,
                                   strides=[1, 1, 1, 1], padding='SAME')
        x = x + self.bias
        # (B, T, C)
        x = tf.squeeze(x, axis=[2])
        return self.dense(x)


class LightConv(tf.keras.layers.Layer):
    """Pay less attention with lightweight and dynamic convolutions

       arxiv: https://arxiv.org/pdf/1901.10430.pdf
    """

    def __init__(self, layer_num, dropout_rate, kernel_size):
        super(LightConv, self).__init__()
        self.attention_blocks = [LightConvBlock(dropout_rate, kernel_size)
                                 for _ in range(layer_num)]

    def call(self, inputs, training=False):
        """inputs contains following two tensor

                   input tensor: shape (B, T, H)
                   input sequence length tensor: shape (B,)
                """
        output_tensor = inputs[0]
        for attention_block in self.attention_blocks:
            output_tensor = attention_block(output_tensor, training=training)
        return output_tensor

