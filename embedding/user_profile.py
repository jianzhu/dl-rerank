import tensorflow as tf

from absl import flags
from tensorflow.keras.layers import DenseFeatures

from embedding.utils import add_mba_reg

FLAGS = flags.FLAGS


class UserProfileEmbedding(tf.keras.layers.Layer):
    """ User profile embedding layer

        transform user profile feature info into embedding representation
    """

    def __init__(self, feature_config, rate=0.3):
        super(UserProfileEmbedding, self).__init__()

        device_spec = tf.DeviceSpec(device_type="CPU", device_index=0)
        with tf.device(device_spec):
            feature_columns = feature_config.get_feature_columns()
            self.ugender_layer = DenseFeatures([feature_columns.get('user.gender')])
            self.uage_layer = DenseFeatures([feature_columns.get('user.age_level')])
            self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, features, training=False):
        device_spec = tf.DeviceSpec(device_type="CPU", device_index=0)
        with tf.device(device_spec):
            # shape: (B, E)
            ugender_emb = self.ugender_layer(features)
            add_mba_reg(self, features, ugender_emb, 'user.gender')
            uage_emb = self.uage_layer(features)
            add_mba_reg(self, features, uage_emb, 'user.age_level')

            # shape: (B, E)
            user_profile_rep = tf.concat([ugender_emb, uage_emb], axis=-1)
            # apply dropout
            user_profile_rep = self.dropout(user_profile_rep, training=training)
            return user_profile_rep
