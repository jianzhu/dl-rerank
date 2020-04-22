import tensorflow as tf

from absl import flags

FLAGS = flags.FLAGS


def add_mba_reg(self, features, embedding, feature_name):
    """  Mini-Batch Aware Regularization

         ref: Deep Interest Network for Click-Through Rate Prediction

         arxiv: https://arxiv.org/abs/1706.06978
    """
    feature = tf.sparse.to_dense(features[feature_name])
    x_flat = tf.reshape(feature, [-1])
    _, unique_idx, unique_count = tf.unique_with_counts(x_flat)
    x_count = tf.map_fn(lambda x: unique_count[x], unique_idx)
    x_count = tf.cast(x_count, tf.float32)
    x_count = tf.reshape(x_count, tf.shape(feature))
    x_count = tf.math.reciprocal(x_count)
    x_count = tf.expand_dims(x_count, axis=-1)
    # add mini-batch aware loss
    self.add_loss(FLAGS.l2_reg_w * tf.reduce_sum(x_count * tf.square(embedding)))
