import tensorflow as tf

from absl import flags
from tensorflow.keras.layers import DenseFeatures

from embedding.utils import add_mba_reg

FLAGS = flags.FLAGS


class ContextEmbedding(tf.keras.layers.Layer):
    """ User profile embedding layer

        transform user profile feature info into embedding representation
    """

    def __init__(self, feature_config, rate=0.3):
        super(ContextEmbedding, self).__init__()

        device_spec = tf.DeviceSpec(device_type="CPU", device_index=0)
        with tf.device(device_spec):
            feature_columns = feature_config.get_feature_columns()
            self.hour_layer = DenseFeatures([feature_columns.get('context.hour')])
            self.phone_layer = DenseFeatures([feature_columns.get('context.phone')])
            self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, features, training=False):
        device_spec = tf.DeviceSpec(device_type="CPU", device_index=0)
        with tf.device(device_spec):
            # shape: (B, E)
            hour_emb = self.hour_layer(features)
            add_mba_reg(self, features, hour_emb, 'context.hour')
            phone_emb = self.phone_layer(features)
            add_mba_reg(self, features, phone_emb, 'context.phone')

            # shape: (B, E)
            context_rep = tf.concat([hour_emb, phone_emb], axis=-1)
            # apply dropout
            context_rep = self.dropout(context_rep, training=training)
            return context_rep
