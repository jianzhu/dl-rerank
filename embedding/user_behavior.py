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
            self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, features, training=False):
        device_spec = tf.DeviceSpec(device_type="CPU", device_index=0)
        with tf.device(device_spec):
            # shape: (B, T, E)
            vgids_emb, _ = self.vgids_layer(features)
            add_mba_reg(self, features, vgids_emb, 'user.visited_goods_ids')
            vsids_emb, _ = self.vsids_layer(features)
            add_mba_reg(self, features, vsids_emb, 'user.visited_shop_ids')
            vcids_emb, _ = self.vcids_layer(features)
            add_mba_reg(self, features, vcids_emb, 'user.visited_cate_ids')
            vgprices_emb, _ = self.vgprices_layer(features)
            add_mba_reg(self, features, vgprices_emb, 'user.visited_goods_prices')

            # shape: (B, T, E)
            user_behavior_rep = tf.concat([vgids_emb, vsids_emb, vcids_emb, vgprices_emb], axis=-1)
            # shape: (B, E)
            user_behavior_rep = tf.reduce_sum(user_behavior_rep, axis=1)
            # apply dropout
            user_behavior_rep = self.dropout(user_behavior_rep, training=training)
            return user_behavior_rep
