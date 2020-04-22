import tensorflow as tf

from absl import flags
from tensorflow.keras.experimental import SequenceFeatures

from embedding.utils import add_mba_reg

FLAGS = flags.FLAGS


class ItemsEmbedding(tf.keras.layers.Layer):
    """ Items embedding layer

        transform items feature info into embedding representation
    """

    def __init__(self, feature_config, rate=0.3):
        super(ItemsEmbedding, self).__init__()

        device_spec = tf.DeviceSpec(device_type="CPU", device_index=0)
        with tf.device(device_spec):
            feature_columns = feature_config.get_feature_columns()
            self.gids_layer = SequenceFeatures([feature_columns.get('item.goods_ids')])
            self.sids_layer = SequenceFeatures([feature_columns.get('item.shop_ids')])
            self.cids_layer = SequenceFeatures([feature_columns.get('item.cate_ids')])
            self.gprices_layer = SequenceFeatures([feature_columns.get('item.goods_prices')])
            self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, features, training=False):
        device_spec = tf.DeviceSpec(device_type="CPU", device_index=0)
        with tf.device(device_spec):
            # shape: (B, T, E)
            gids_emb, _ = self.gids_layer(features)
            add_mba_reg(self, features, gids_emb, 'item.goods_ids')
            sids_emb, _ = self.sids_layer(features)
            add_mba_reg(self, features, sids_emb, 'item.shop_ids')
            cids_emb, _ = self.cids_layer(features)
            add_mba_reg(self, features, cids_emb, 'item.cate_ids')
            gprices_emb, _ = self.gprices_layer(features)
            add_mba_reg(self, features, gprices_emb, 'item.goods_prices')

            # shape: (B, T, E)
            items_rep = tf.concat([gids_emb, sids_emb, cids_emb, gprices_emb], axis=-1)
            # apply dropout
            items_rep = self.dropout(items_rep, training=training)
            return items_rep
