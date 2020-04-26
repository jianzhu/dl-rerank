import tensorflow as tf

from modeling.activations import dice


class DIN(tf.keras.layers.Layer):
    """ Deep interest network attention

       ref: Deep Interest Network for Click-Through Rate Prediction

       arxiv: https://arxiv.org/abs/1706.06978
    """

    def __init__(self, dropout_rate):
        super(DIN, self).__init__()

        # attention dense layer
        self.dense1 = tf.keras.layers.Dense(units=36)
        self.dice = dice.Dice()
        self.dense2 = tf.keras.layers.Dense(units=1)

        # multi-layer nonlinear transformation
        self.mlp_bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.mlp_drop1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.mlp_dense1 = tf.keras.layers.Dense(units=256, activation='relu')
        self.mlp_bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.mlp_drop2 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.mlp_dense2 = tf.keras.layers.Dense(units=128, activation='relu')
        self.mlp_bn3 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.mlp_drop3 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.mlp_dense3 = tf.keras.layers.Dense(units=64, activation='relu')

    def mlp(self, interest, user_profile, context, iseq_len, training=False):
        """do multi-layer nonlinear transformation"""
        # user profile info
        # shape: (B, E)
        user_profile = tf.expand_dims(user_profile, axis=1)
        # shape: (B, T', E)
        user_profile = tf.tile(user_profile, [1, iseq_len, 1])

        # context info
        # shape: (B, E)
        context = tf.expand_dims(context, axis=1)
        # shape: (B, T', E)
        context = tf.tile(context, [1, iseq_len, 1])

        x = tf.concat([interest, user_profile, context], axis=-1)
        x = self.mlp_bn1(x, training=training)
        x = self.mlp_drop1(x, training=training)
        x = self.mlp_dense1(x, training=training)
        x = self.mlp_bn2(x, training=training)
        x = self.mlp_drop2(x, training=training)
        x = self.mlp_dense2(x, training=training)
        x = self.mlp_bn3(x, training=training)
        x = self.mlp_drop3(x, training=training)
        # (B, T', 64)
        return self.mlp_dense3(x, training=training)

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
        user_profile = inputs[2]
        context = inputs[3]

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
        query = tf.expand_dims(user_behavior, axis=1)
        # (B, T', T, E)
        query = tf.tile(query, [1, iseq_len, 1, 1])

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
        key = tf.expand_dims(items, axis=2)
        # (B, T', T, E)
        key = tf.tile(key, [1, 1, useq_len, 1])

        # (B, T', T, 4E)
        x = tf.concat([query, key, query - key, query * key], axis=-1)
        # (B, T', T, 36)
        x = self.dense1(x)
        # (B, T', T, 36)
        x = self.dice(x, training=training)
        # (B, T', T, 1)
        x = self.dense2(x)
        # (B, T', T)
        weight = tf.squeeze(x, [3])
        # (B, T', E) = (B, T', T) x (B, T, E)
        user_interest = tf.matmul(weight, user_behavior)
        # (B, T', 64)
        return self.mlp(user_interest, user_profile, context, iseq_len, training)
