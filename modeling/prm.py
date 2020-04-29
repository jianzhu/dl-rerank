import tensorflow as tf

from absl import flags

from embedding.items import ItemsEmbedding
from embedding.user_profile import UserProfileEmbedding
from embedding.user_behavior import UserBehaviorEmbedding
from embedding.context import ContextEmbedding

from modeling.attentions.din import DIN
from modeling.attentions.transformer import Transformer
from modeling.attentions.light_conv import LightConv

FLAGS = flags.FLAGS


class PRM(tf.keras.layers.Layer):
    """ Personalized ReRanker

    ref: Personalized Re-ranking for Recommendation
         arxiv: https://arxiv.org/abs/1904.06813
    """

    def __init__(self, feature_config):
        super(PRM, self).__init__()
        # items embedding
        self.items_emb = ItemsEmbedding(feature_config, rate=FLAGS.dropout_rate)
        # user profile embedding
        self.user_profile_emb = UserProfileEmbedding(feature_config, rate=FLAGS.dropout_rate)
        # user behavior embedding
        self.user_behavior_emb = UserBehaviorEmbedding(feature_config, rate=FLAGS.dropout_rate)
        # context embedding
        self.context_emb = ContextEmbedding(feature_config, rate=FLAGS.dropout_rate)

        # din: modeling user interest
        self.din = DIN(dropout_rate=FLAGS.dropout_rate)

        # self attention: modeling <user, item> self interaction
        if FLAGS.self_att_type == 'transformer':
            self.self_attention = Transformer(layer_num=FLAGS.layer_num,
                                              head_num=FLAGS.head_num,
                                              hidden_size=FLAGS.hidden_size,
                                              filter_size=FLAGS.filter_size,
                                              dropout_rate=FLAGS.dropout_rate)
        elif FLAGS.self_att_type == 'light_conv':
            self.self_attention = LightConv(layer_num=FLAGS.layer_num,
                                            dropout_rate=FLAGS.dropout_rate,
                                            kernel_size=FLAGS.kernel_size)

        # embedding mlp transformation
        self.mlp_emb_bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.mlp_emb_drop1 = tf.keras.layers.Dropout(rate=FLAGS.dropout_rate)
        self.mlp_emb_dense1 = tf.keras.layers.Dense(64)
        self.mlp_emb_bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.mlp_emb_drop2 = tf.keras.layers.Dropout(rate=FLAGS.dropout_rate)
        self.mlp_emb_dense2 = tf.keras.layers.Dense(FLAGS.hidden_size)

        # output mlp transformation
        self.mlp_bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.mlp_drop1 = tf.keras.layers.Dropout(rate=FLAGS.dropout_rate)
        self.mlp_dense1 = tf.keras.layers.Dense(units=200, activation='relu')
        self.mlp_bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.mlp_drop2 = tf.keras.layers.Dropout(rate=FLAGS.dropout_rate)
        self.mlp_dense2 = tf.keras.layers.Dense(units=80, activation='relu')
        self.mlp_bn3 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.mlp_drop3 = tf.keras.layers.Dropout(rate=FLAGS.dropout_rate)
        self.mlp_dense3 = tf.keras.layers.Dense(units=1)

    def mlp_emb(self, inputs, training=False):
        outputs = self.mlp_emb_bn1(inputs, training=training)
        outputs = self.mlp_emb_drop1(outputs, training=training)
        outputs = self.mlp_emb_dense1(outputs, training=training)
        outputs = self.mlp_emb_bn2(outputs, training=training)
        outputs = self.mlp_emb_drop2(outputs, training=training)
        return self.mlp_emb_dense2(outputs, training=training)

    def mlp(self, outputs, training=False):
        outputs = self.mlp_bn1(outputs, training=training)
        outputs = self.mlp_drop1(outputs, training=training)
        outputs = self.mlp_dense1(outputs, training=training)
        outputs = self.mlp_bn2(outputs, training=training)
        outputs = self.mlp_drop2(outputs, training=training)
        outputs = self.mlp_dense2(outputs, training=training)
        outputs = self.mlp_bn3(outputs, training=training)
        outputs = self.mlp_drop3(outputs, training=training)
        outputs = self.mlp_dense3(outputs, training=training)
        return outputs

    def call(self, features, training=False):
        items = self.items_emb(features, training=training)
        user_profile = self.user_profile_emb(features, training=training)
        user_behavior = self.user_behavior_emb(features, training=training)
        context = self.context_emb(features, training=training)

        # user interest info
        # shape: (B, T, E)
        personal_rep = self.din([user_behavior[0], items[0], user_profile, context], training=training)
        # shape: (B, T, 32)
        inputs = self.mlp_emb(tf.concat([personal_rep, items[0]], axis=-1), training=training)

        # do self-attention
        outputs = self.self_attention([inputs, items[1]], training=training)

        # do mlp transformation
        outputs = self.mlp(outputs, training=training)

        # logits (B, T, 1), input_seq_mask (B, T)
        return [outputs, tf.sequence_mask(items[1])]
