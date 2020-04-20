import tensorflow as tf


class ItemsEmbedding(tf.keras.layers.Layer):
    """ Items embedding layer

        transform items feature info into embedding representation
    """

    def __init__(self, feature_config, rate=0.3):
        super(ItemsEmbedding, self).__init__()

        feature_columns = feature_config.get_feature_columns()
        columns = [feature_columns.get('item.goods_ids'),
                   feature_columns.get('item.shop_ids'),
                   feature_columns.get('item.cate_ids')]
        self.items_layer = tf.keras.experimental.SequenceFeatures(columns)
        self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, features, training=False):
        # shape: (B, T, E)
        sequence_embed = self.items_layer(features)
        # shape: (B, T, 1)
        sequence_price = tf.sparse.to_dense(features['item.goods_prices'])
        # shape: (B, T, E+1)
        items_rep = tf.concat([sequence_embed, sequence_price], axis=-1)
        # apply dropout
        items_rep = self.dropout(items_rep, training=training)
        return items_rep
