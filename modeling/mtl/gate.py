import tensorflow as tf


class Gate(tf.keras.layers.Layer):
    """Gate layer on top of Shared Bottom Layer and Expert

       ref: Recommending What Video to Watch Next: A Multitask Ranking System
    """

    def __init__(self, expert_num, gate_dropout):
        super(Gate, self).__init__()

        self.dense = tf.keras.layers.Dense(units=expert_num, activation='softmax')
        self.dropout = tf.keras.layers.Dropout(rate=gate_dropout)

    def call(self, inputs, training=False):
        shared_bottom = inputs['shared_bottom']
        experts = inputs['experts']

        # shape: (B, T, expert_num)
        weights = self.dense(shared_bottom)
        # shape: (B, T, 1, expert_num)
        weights = tf.expand_dims(weights, axis=-2)

        # shape: (B, T, expert_num, H)
        x = tf.stack(experts, axis=2)
        # shape: (B, T, 1, H)
        x = tf.matmul(weights, x)
        # shape: (B, T, H)
        return tf.squeeze(x, axis=[2])
