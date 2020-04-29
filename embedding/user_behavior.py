import tensorflow as tf

from absl import flags
from tensorflow.keras.experimental import SequenceFeatures

from embedding.utils import add_mba_reg

FLAGS = flags.FLAGS


class UserBehaviorEmbedding(tf.keras.layers.Layer):
    """ User behavior embedding layer

        transform user behavior feature info into embedding representation
    """

    def __init__(self, feature_config, rate=0.3):
        super(UserBehaviorEmbedding, self).__init__()

        device_spec = tf.DeviceSpec(device_type="CPU", device_index=0)
        with tf.device(device_spec):
            feature_columns = feature_config.get_feature_columns()
            self.vgids_layer = SequenceFeatures([feature_columns.get('user.visited_goods_ids')])
            self.vsids_layer = SequenceFeatures([feature_columns.get('user.visited_shop_ids')])
            self.vcids_layer = SequenceFeatures([feature_columns.get('user.visited_cate_ids')])
            self.vgprices_layer = SequenceFeatures([feature_columns.get('user.visited_goods_prices')])

            # multi-layer projection
            self.mlp_bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
            self.mlp_drop1 = tf.keras.layers.Dropout(rate=rate)
            self.mlp_dense1 = tf.keras.layers.Dense(256, activation='relu')
            self.mlp_bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
            self.mlp_drop2 = tf.keras.layers.Dropout(rate=rate)
            self.mlp_dense2 = tf.keras.layers.Dense(128, activation='relu')
            self.mlp_bn3 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
            self.mlp_drop3 = tf.keras.layers.Dropout(rate=rate)
            self.mlp_dense3 = tf.keras.layers.Dense(FLAGS.hidden_size, activation='relu')

    def mlp(self, inputs, training=False):
        x = self.mlp_bn1(inputs, training=training)
        x = self.mlp_drop1(x, training=training)
        x = self.mlp_dense1(x)
        x = self.mlp_bn2(x, training=training)
        x = self.mlp_drop2(x, training=training)
        x = self.mlp_dense2(x)
        x = self.mlp_bn3(x, training=training)
        x = self.mlp_drop3(x, training=training)
        return self.mlp_dense3(x)

    def call(self, features, training=False):
        device_spec = tf.DeviceSpec(device_type="CPU", device_index=0)

        with tf.device(device_spec):
            # shape: (B, T, E)
            vgids_emb, sequence_len = self.vgids_layer(features)
            vsids_emb, _ = self.vsids_layer(features)
            vcids_emb, _ = self.vcids_layer(features)
            vgprices_emb, _ = self.vgprices_layer(features)

            if training:
                add_mba_reg(self, features, vgids_emb, 'user.visited_goods_ids')
                add_mba_reg(self, features, vsids_emb, 'user.visited_shop_ids')
                add_mba_reg(self, features, vcids_emb, 'user.visited_cate_ids')
                add_mba_reg(self, features, vgprices_emb, 'user.visited_goods_prices')

            # shape: (B, T, E)
            user_behavior_rep = tf.concat([vgids_emb, vsids_emb, vcids_emb, vgprices_emb], axis=-1)
            # shape: (B, T, 64)
            user_behavior_rep = self.mlp(user_behavior_rep, training=training)
            return [user_behavior_rep, sequence_len]
