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
                   feature_columns.get('item.cate_ids'),
                   feature_columns.get('item.goods_prices')]
        self.items_layer = tf.keras.experimental.SequenceFeatures(columns)
        self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, features, training=False):
        # shape: (B, T, E)
        items_rep, _ = self.items_layer(features)
        # apply dropout
        items_rep = self.dropout(items_rep, training=training)
        return items_rep
