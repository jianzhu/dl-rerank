import tensorflow as tf

from absl import flags

from embedding.items import ItemsEmbedding
from embedding.user_profile import UserProfileEmbedding
from embedding.user_behavior import UserBehaviorEmbedding
from embedding.context import ContextEmbedding

from modeling.attentions.din import DIN
from modeling.attentions.transformer import Transformer
from modeling.attentions.light_conv import LightConv
from modeling.attentions.lite_transformer import LiteTransformer
from modeling.mtl.tasks import Tasks

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
        elif FLAGS.self_att_type == 'lite_transformer':
            self.self_attention = LiteTransformer(layer_num=FLAGS.layer_num,
                                                  head_num=FLAGS.head_num,
                                                  hidden_size=FLAGS.hidden_size,
                                                  filter_size=FLAGS.filter_size,
                                                  kernel_size=FLAGS.kernel_size,
                                                  dropout_rate=FLAGS.dropout_rate)
        else:
            raise ValueError('invalid attention type: %s' % FLAGS.self_att_type)

        # embedding mlp transformation
        self.mlp_emb_bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.mlp_emb_drop1 = tf.keras.layers.Dropout(rate=FLAGS.dropout_rate)
        self.mlp_emb_dense1 = tf.keras.layers.Dense(64)
        self.mlp_emb_bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.mlp_emb_drop2 = tf.keras.layers.Dropout(rate=FLAGS.dropout_rate)
        self.mlp_emb_dense2 = tf.keras.layers.Dense(FLAGS.hidden_size)

        # multi-task learning
        self.tasks = Tasks(FLAGS.config_dir, FLAGS.dropout_rate, FLAGS.gate_dropout_rate)

    def mlp_emb(self, inputs, training=False):
        outputs = self.mlp_emb_bn1(inputs, training=training)
        outputs = self.mlp_emb_drop1(outputs, training=training)
        outputs = self.mlp_emb_dense1(outputs, training=training)
        outputs = self.mlp_emb_bn2(outputs, training=training)
        outputs = self.mlp_emb_drop2(outputs, training=training)
        return self.mlp_emb_dense2(outputs, training=training)

    def call(self, inputs, training=False):
        features = inputs[0]
        labels = inputs[1]

        items = self.items_emb(features, training=training)
        user_profile = self.user_profile_emb(features, training=training)
        user_behavior = self.user_behavior_emb(features, training=training)
        context = self.context_emb(features, training=training)
        # user interest info
        # shape: (B, T, E)
        personal_rep = self.din([user_behavior[0], items[0], user_profile, context], training=training)
        # shape: (B, T, E')
        inputs = self.mlp_emb(tf.concat([personal_rep, items[0]], axis=-1), training=training)
        # do self-attention
        shared_bottom = self.self_attention([inputs, items[1]], training=training)
        # do multi-task learning
        inputs = [shared_bottom, tf.sequence_mask(items[1]), items[2], labels]
        return self.tasks(inputs)
