import tensorflow as tf

from absl import flags
from tensorflow.keras.experimental import SequenceFeatures

from embedding.utils import add_mba_reg
from embedding.text.conv_kernel import QueryTextConv

FLAGS = flags.FLAGS


class UserBehaviorEmbedding(tf.keras.layers.Layer):
    """ User behavior embedding layer

        transform user behavior feature info into embedding representation
    """

    def __init__(self, feature_config, rate=0.3):
        super(UserBehaviorEmbedding, self).__init__()

        feature_configs = feature_config.get_feature_configs()
        self.query_len = feature_configs['user.query_word_ids']['query_len']
        device_spec = tf.DeviceSpec(device_type="CPU", device_index=0)
        with tf.device(device_spec):
            feature_columns = feature_config.get_feature_columns()
            self.vgids_layer = SequenceFeatures([feature_columns.get('user.visited_goods_ids')])
            self.vsids_layer = SequenceFeatures([feature_columns.get('user.visited_shop_ids')])
            self.vcids_layer = SequenceFeatures([feature_columns.get('user.visited_cate_ids')])
            self.vgprices_layer = SequenceFeatures([feature_columns.get('user.visited_goods_prices')])
            self.query_layer = SequenceFeatures([feature_columns.get('user.query_word_ids')])

        # item text convolution layer
        self.query_conv_layer = QueryTextConv(FLAGS.qtxt_filters, FLAGS.qtxt_kernel_sizes, self.query_len)
        # multi-layer projection
        self.mlp_bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.mlp_drop1 = tf.keras.layers.Dropout(rate=rate)
        self.mlp_dense1 = tf.keras.layers.Dense(FLAGS.be_filter_size, activation='relu')
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
            vgids_emb, sequence_len = self.vgids_layer(features)
            vsids_emb, _ = self.vsids_layer(features)
            vcids_emb, _ = self.vcids_layer(features)
            vgprices_emb, _ = self.vgprices_layer(features)

            if training:
                add_mba_reg(self, features, vgids_emb, 'user.visited_goods_ids')
                add_mba_reg(self, features, vsids_emb, 'user.visited_shop_ids')
                add_mba_reg(self, features, vcids_emb, 'user.visited_cate_ids')
                add_mba_reg(self, features, vgprices_emb, 'user.visited_goods_prices')

            vgoods_shape = tf.shape(vgids_emb)
            query_emb = self.text_emb(features, self.query_layer, self.query_conv_layer, vgoods_shape[1])
            # shape: (B, T, E)
            user_behavior_rep = tf.concat([vgids_emb, vsids_emb, vcids_emb, vgprices_emb, query_emb], axis=-1)
            # shape: (B, T, 64)
            user_behavior_rep = self.mlp(user_behavior_rep, training=training)
            return [user_behavior_rep, sequence_len]

    def text_emb(self, features, text_emb_layer, text_conv_layer, item_num):
        # text emb shape: (B, W, H)
        # B - batch size
        # W - query word id num
        # H - word id embedding size
        text_emb, _ = text_emb_layer(features)
        # text emb shape: (B, E)
        text_emb = text_conv_layer(text_emb)
        # text emb shape: (B, T, E)
        return tf.tile(text_emb, [1, item_num, 1])
