import tensorflow as tf


class Dice(tf.keras.layers.Layer):
    """Dice activation function layer

       ref: Deep Interest Network for Click-Through Rate Prediction

       arxiv: https://arxiv.org/abs/1706.06978
    """

    def __init__(self):
        super(Dice, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization(axis=-1, center=True, scale=True, epsilon=1e-7)

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(input_shape[-1],),
                                     initializer='random_normal',
                                     trainable=True)

    def call(self, inputs, training=False):
        x = self.bn(inputs, training)
        p = tf.sigmoid(x)
        return p * inputs + (1 - p) * inputs * self.alpha
