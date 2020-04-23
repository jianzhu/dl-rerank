import tensorflow as tf

from modeling.activations import dice


class DIN(tf.keras.layers.Layer):
    """ Deep interest network attention

       ref: Deep Interest Network for Click-Through Rate Prediction

       arxiv: https://arxiv.org/abs/1706.06978
    """

    def __init__(self):
        super(DIN, self).__init__()

        self.dense1 = tf.keras.layers.Dense(units=36)
        self.dice = dice.Dice()
        self.dense2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=False):
        """
        input:
            user_behavior: shape (B, T, E)
            items: shape (B, T', E)

        output:
            weighted user_behavior: shape (B, T', E)
        """
        user_behavior = inputs[0]
        items = inputs[1]

        # expand user behavior dim then broadcast to item num
        # after broadcast, user_info's shape is (B, T', T, E)
        # each dimension meaning as following
        # B  -- batch size
        # T' -- target item num
        # each item's corresponding user behavior matrix's shape is (T, E)
        # T  -- user behavior num
        # E  -- each behavior vector size
        iseq_len = tf.shape(items)[1]
        # (B, 1, T, E)
        user_info = tf.expand_dims(user_behavior, axis=1)
        # (B, T', T, E)
        user_info = tf.tile(user_info, [1, iseq_len, 1, 1])

        # expand item dim then broadcast to user behavior num
        # after broadcast, item_info's shape is (B, T', T, E)
        # each dimension meaning as following
        # B  -- batch size
        # T' -- target item num
        # each item's corresponding user behavior matrix's shape is (T, E)
        # T  -- user behavior num
        # E  -- each item vector size
        useq_len = tf.shape(user_behavior)[1]
        # (B, T', 1, E)
        item_info = tf.expand_dims(items, axis=2)
        # (B, T', T, E)
        item_info = tf.tile(item_info, [1, 1, useq_len, 1])

        # (B, T', T, 4E)
        x = tf.concat([user_info, item_info, user_info - item_info, user_info * item_info], axis=-1)
        # (B, T', T, 36)
        x = self.dense1(x)
        # (B, T', T, 36)
        x = self.dice(x, training=training)
        x = self.dense2(x)
        # (B, T', T)
        weight = tf.squeeze(x, [3])
        # (B, T', E) = (B, T', T) x (B, T, E)
        return tf.matmul(weight, user_behavior)
