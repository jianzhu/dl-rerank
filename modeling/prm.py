import tensorflow as tf

from embedding.items import ItemsEmbedding
from embedding.user_profile import UserProfileEmbedding
from embedding.user_behavior import UserBehaviorEmbedding
from embedding.context import ContextEmbedding

from modeling.attentions.din import DIN
from modeling.attentions.transformer import Transformer


class PRM(tf.keras.layers.Layer):
    """ Personalized ReRanker

    ref: Personalized Re-ranking for Recommendation
         arxiv: https://arxiv.org/abs/1904.06813
    """

    def __init__(self,
                 feature_config,
                 layer_num,
                 head_num,
                 hidden_size,
                 filter_size,
                 dropout_rate=0.3):
        super(PRM, self).__init__()
        # items embedding
        self.items_emb = ItemsEmbedding(feature_config, rate=dropout_rate)
        # user profile embedding
        self.user_profile_emb = UserProfileEmbedding(feature_config, rate=dropout_rate)
        # user behavior embedding
        self.user_behavior_emb = UserBehaviorEmbedding(feature_config, rate=dropout_rate)
        # context embedding
        self.context_emb = ContextEmbedding(feature_config, rate=dropout_rate)

        # din: modeling user interest
        self.din = DIN()

        # transformer: modeling <user, item> self interaction
        self.transformer = Transformer(layer_num=layer_num,
                                       head_num=head_num,
                                       hidden_size=hidden_size,
                                       filter_size=filter_size,
                                       dropout_rate=dropout_rate)

        # output mlp transformation
        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dense1 = tf.keras.layers.Dense(units=200, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.drop2 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dense2 = tf.keras.layers.Dense(units=80, activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.drop3 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dense3 = tf.keras.layers.Dense(units=1)

    def mlp(self, outputs, training=False):
        outputs = self.bn1(outputs, training=training)
        outputs = self.drop1(outputs, training=training)
        outputs = self.bn2(outputs, training=training)
        outputs = self.drop2(outputs, training=training)
        outputs = self.bn3(outputs, training=training)
        outputs = self.drop3(outputs, training=training)
        outputs = self.dense3(outputs, training=training)
        return outputs

    def call(self, features, training=False):
        items_info = self.items_emb(features, training=training)
        user_info = self.user_profile_emb(features, training=training)
        behavior_info = self.user_behavior_emb(features, training=training)
        context_info = self.context_emb(features, training=training)

        seq_len = tf.shape(items_info[0])[1]

        # user profile info
        # shape: (B, E)
        user_info = tf.expand_dims(user_info, axis=1)
        # shape: (B, T, E)
        user_info = tf.tile(user_info, [1, seq_len, 1])

        # context info
        # shape: (B, E)
        context_info = tf.expand_dims(context_info, axis=1)
        # shape: (B, T, E)
        context_info = tf.tile(context_info, [1, seq_len, 1])

        # user interest info
        # shape: (B, T, E)
        interest_info = self.din([behavior_info[0], items_info[0]], training=training)
        # shape: (B, T, E)
        outputs = tf.concat([user_info, context_info, interest_info, items_info[0]], axis=-1)

        # do self-attention
        outputs = self.transformer([outputs, items_info[1]], training=training)

        # do mlp transformation
        outputs = self.mlp(outputs, training=training)

        # logits (B, T, 1), input_seq_mask (B, T)
        return [outputs, tf.sequence_mask(items_info[1])]
