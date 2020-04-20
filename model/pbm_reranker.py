import tensorflow as tf

from embedding.items import ItemsEmbedding


class PBMReRanker(tf.keras.models.Model):
    """ Personalized ReRanker

    ref: Personalized Re-ranking for Recommendation
         arxiv: https://arxiv.org/abs/1904.06813
    """

    def __init__(self, feature_config, rate=0.3):
        super(PBMReRanker, self).__init__()
        # items embedding
        self.items_emb = ItemsEmbedding(feature_config, rate)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, features, training=False):
        x = self.items_emb(features, training=training)
        x = self.batch_norm(x, training=training)
        logits = self.dense(x)
        return logits
