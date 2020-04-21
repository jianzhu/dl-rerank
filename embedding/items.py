import tensorflow as tf

from absl import flags
from tensorflow.keras.experimental import SequenceFeatures

FLAGS = flags.FLAGS


class ItemsEmbedding(tf.keras.layers.Layer):
    """ Items embedding layer

        transform items feature info into embedding representation
    """

    def __init__(self, feature_config, rate=0.3):
        super(ItemsEmbedding, self).__init__()

        feature_columns = feature_config.get_feature_columns()
        self.goods_ids_layer = SequenceFeatures([feature_columns.get('item.goods_ids')])
        self.shop_ids_layer = SequenceFeatures([feature_columns.get('item.shop_ids')])
        self.cate_ids_layer = SequenceFeatures([feature_columns.get('item.cate_ids')])
        self.goods_prices_layer = SequenceFeatures([feature_columns.get('item.goods_prices')])
        self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, features, training=False):
        # shape: (B, T, E)
        goods_ids_emb, _ = self.goods_ids_layer(features)
        self.add_mba_reg(features, goods_ids_emb, 'item.goods_ids')
        shop_ids_emb, _ = self.shop_ids_layer(features)
        self.add_mba_reg(features, shop_ids_emb, 'item.shop_ids')
        cate_ids_emb, _ = self.cate_ids_layer(features)
        self.add_mba_reg(features, cate_ids_emb, 'item.cate_ids')
        goods_prices_emb, _ = self.goods_prices_layer(features)
        self.add_mba_reg(features, goods_prices_emb, 'item.goods_prices')

        # shape: (B, T, E)
        items_rep = tf.concat([goods_ids_emb, shop_ids_emb,
                               cate_ids_emb, goods_prices_emb], axis=-1)
        # apply dropout
        items_rep = self.dropout(items_rep, training=training)
        return items_rep

    def add_mba_reg(self, features, embedding, feature_name):
        # shape: (B, T)
        feature = tf.sparse.to_dense(features[feature_name])
        x_flat = tf.reshape(feature, [-1])
        _, unique_idx, unique_count = tf.unique_with_counts(x_flat)
        x_count = tf.map_fn(lambda x: unique_count[x], unique_idx)
        x_count = tf.cast(x_count, tf.float32)
        x_count = tf.reshape(x_count, tf.shape(feature))
        x_count = tf.math.reciprocal(x_count)
        # shape: (B, T, 1)
        x_count = tf.expand_dims(x_count, axis=-1)
        # add mini-batch aware loss
        self.add_loss(FLAGS.l2_reg_w * tf.reduce_sum(x_count * tf.square(embedding)))
