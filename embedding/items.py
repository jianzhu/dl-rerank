import tensorflow as tf

from absl import flags
from tensorflow.keras.experimental import SequenceFeatures

from embedding.utils import add_mba_reg
from embedding.text.conv_kernel import ItemsTextConv

FLAGS = flags.FLAGS


class ItemsEmbedding(tf.keras.layers.Layer):
    """ Items embedding layer

        transform items feature info into embedding representation
    """

    def __init__(self, feature_config, rate=0.3):
        super(ItemsEmbedding, self).__init__()

        feature_configs = feature_config.get_feature_configs()
        embedding_configs = feature_config.get_embedding_configs()
        self.title_len = feature_configs['item.title_word_ids']['title_len']
        self.twe_dim = embedding_configs['item.title_word_ids']
        self.content_len = feature_configs['item.content_word_ids']['content_len']
        self.cwe_dim = embedding_configs['item.content_word_ids']
        device_spec = tf.DeviceSpec(device_type="CPU", device_index=0)
        with tf.device(device_spec):
            feature_columns = feature_config.get_feature_columns()
            self.gids_layer = SequenceFeatures([feature_columns.get('item.goods_ids')])
            self.sids_layer = SequenceFeatures([feature_columns.get('item.shop_ids')])
            self.cids_layer = SequenceFeatures([feature_columns.get('item.cate_ids')])
            self.gprices_layer = SequenceFeatures([feature_columns.get('item.goods_prices')])
            self.rankpos_layer = SequenceFeatures([feature_columns.get('item.rank_pos')])
            self.showpos_layer = SequenceFeatures([feature_columns.get('item.show_pos')])
            self.title_layer = SequenceFeatures([feature_columns.get('item.title_word_ids')])
            self.content_layer = SequenceFeatures([feature_columns.get('item.content_word_ids')])

        # item text convolution layer
        self.title_conv_layer = ItemsTextConv(FLAGS.ttxt_filters, FLAGS.ttxt_kernel_sizes, self.title_len)
        self.content_conv_layer = ItemsTextConv(FLAGS.ctxt_filters, FLAGS.ctxt_kernel_sizes, self.content_len)

        # multi-layer projection
        self.mlp_bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.mlp_drop1 = tf.keras.layers.Dropout(rate=rate)
        self.mlp_dense1 = tf.keras.layers.Dense(FLAGS.ie_filter_size, activation='relu')
        self.mlp_bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.mlp_drop2 = tf.keras.layers.Dropout(rate=rate)
        self.mlp_dense2 = tf.keras.layers.Dense(FLAGS.hidden_size, activation='relu')

    def mlp(self, inputs, training=False):
        x = self.mlp_bn1(inputs, training=training)
        x = self.mlp_drop1(x, training=training)
        x = self.mlp_dense1(x)
        x = self.mlp_bn2(x, training=training)
        x = self.mlp_drop2(x, training=training)
        return self.mlp_dense2(x)

    def call(self, features, training=False):
        device_spec = tf.DeviceSpec(device_type="CPU", device_index=0)
        with tf.device(device_spec):
            # shape: (B, T, E)
            gids_emb, sequence_len = self.gids_layer(features)
            sids_emb, _ = self.sids_layer(features)
            cids_emb, _ = self.cids_layer(features)
            gprices_emb, _ = self.gprices_layer(features)
            rankpos_emb, _ = self.rankpos_layer(features)
            showpos_emb, _ = self.showpos_layer(features)
            if training:
                add_mba_reg(self, features, gids_emb, 'item.goods_ids')
                add_mba_reg(self, features, sids_emb, 'item.shop_ids')
                add_mba_reg(self, features, cids_emb, 'item.cate_ids')
                add_mba_reg(self, features, gprices_emb, 'item.goods_prices')

            title_emb = self.text_emb(features,
                                      self.title_layer,
                                      self.title_conv_layer,
                                      self.title_len,
                                      self.twe_dim)
            content_emb = self.text_emb(features,
                                        self.content_layer,
                                        self.content_conv_layer,
                                        self.content_len,
                                        self.cwe_dim)
            # shape: (B, T, E)
            items_rep = tf.concat([gids_emb, sids_emb,
                                   cids_emb, gprices_emb, title_emb, content_emb], axis=-1)
            # modeling rank pos
            items_rep = rankpos_emb + items_rep
            # shape: (B, T, 64)
            items_rep = self.mlp(items_rep, training=training)
            return [items_rep, sequence_len, showpos_emb]

    def text_emb(self, features, text_emb_layer, text_conv_layer, word_num, embedding_dim):
        # text emb shape: (B, T x W, H)
        # B - batch size
        # T - item num
        # W - item word id num
        # H - word id embedding size
        text_emb, _ = text_emb_layer(features)
        batch_size = tf.shape(text_emb)[0]
        # text emb shape: (B, T, W, H)
        text_emb = tf.reshape(text_emb, shape=(batch_size, -1, word_num, embedding_dim))
        # text emb shape: (B, T, E)
        return text_conv_layer(text_emb)
