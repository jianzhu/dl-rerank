import tensorflow as tf

from embedding.items import ItemsEmbedding
from embedding.user_profile import UserProfileEmbedding
from embedding.user_behavior import UserBehaviorEmbedding
from embedding.context import ContextEmbedding


class PBMReRanker(tf.keras.models.Model):
    """ Personalized ReRanker

    ref: Personalized Re-ranking for Recommendation
         arxiv: https://arxiv.org/abs/1904.06813
    """

    def __init__(self, feature_config, rate=0.3):
        super(PBMReRanker, self).__init__()
        # items embedding
        self.items_emb = ItemsEmbedding(feature_config, rate)
        # user profile embedding
        self.user_profile_emb = UserProfileEmbedding(feature_config, rate)
        # user behavior embedding
        self.user_behavior_emb = UserBehaviorEmbedding(feature_config, rate)
        # context embedding
        self.context_emb = ContextEmbedding(feature_config, rate)

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(1, dtype=tf.float32)

    def call(self, features, training=False):
        # item info
        # shape: (B, T, E)
        item_rep = self.items_emb(features, training=training)
        seq_len = tf.shape(item_rep)[1]

        # user profile info
        # shape: (B, E)
        user_profile_rep = self.user_profile_emb(features, training=training)
        user_profile_rep = tf.expand_dims(user_profile_rep, axis=1)
        # shape: (B, T, E)
        user_profile_rep = tf.tile(user_profile_rep, [1, seq_len, 1])

        # user behavior info
        # shape: (B, E)
        user_behavior_rep = self.user_behavior_emb(features, training=training)
        user_behavior_rep = tf.expand_dims(user_behavior_rep, axis=1)
        # shape: (B, T, E)
        user_behavior_rep = tf.tile(user_behavior_rep, [1, seq_len, 1])

        # context info
        # shape: (B, E)
        context_rep = self.context_emb(features, training=training)
        context_rep = tf.expand_dims(context_rep, axis=1)
        # shape: (B, T, E)
        context_rep = tf.tile(context_rep, [1, seq_len, 1])

        x = tf.concat([user_profile_rep, user_behavior_rep, item_rep, context_rep], axis=-1)
        x = self.batch_norm(x, training=training)
        logits = self.dense(x)
        return logits
