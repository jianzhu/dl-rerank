import tensorflow as tf
from tensorflow import keras


class DINAttention(keras.layers.Layer):
    """Deep Interest Network Attention


    """
    def __init__(self, dropout_rate, behavior_input):
        super(DINAttention, self).__init__()
        self.drop1 = keras.layers.Dropout(rate=dropout_rate)
        self.bn1 = keras.layers.BatchNormalization()
        self.dense1 = keras.layers.Dense(units=36, activation='relu')
        self.drop2 = keras.layers.Dropout(rate=dropout_rate)
        self.bn2 = keras.layers.BatchNormalization()
        self.dense = keras.layers.Dense(units=1)
        self.behavior_input = behavior_input

    def call(self, ad_input, training=False):
        expand_ad_input = tf.expand_dims(ad_input, 1)
        seq_len = tf.shape(self.behavior_input)[1]
        expand_ad_input = tf.tile(expand_ad_input, [1, seq_len, 1])
        x = tf.concat([expand_ad_input, self.behavior_input], axis=-1)
        x = self.drop1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.dense1(x)
        x = self.drop2(x, training=training)
        x = self.bn2(x, training=training)
        weight = self.dense(x)
        output = weight * self.behavior_input
        output = tf.reduce_sum(output, axis=1)
        return output


import numpy as np

x = np.random.rand(2, 3, 3)
y = np.random.rand(2, 2, 2)

attention = DINAttention(0.2, x)

print('y')
print('-' * 40)
print(y)
print('-' * 40)

attention = keras.layers.TimeDistributed(attention)
result = attention(y, training=True)
print('result')
print('-' * 40)
print(result)
print('-' * 40)
