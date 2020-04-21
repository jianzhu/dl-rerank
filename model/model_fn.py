import tensorflow as tf
import tensorflow_addons as tfa

from absl import flags

from model.pbm_reranker import PBMReRanker

FLAGS = flags.FLAGS


def model_fn(features, labels, mode, params):
    feature_config = params['feature_config']
    dropout_rate = params['dropout_rate']
    pbm_reranker = PBMReRanker(feature_config, rate=dropout_rate)

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = pbm_reranker(features, training)
        prediction = tf.nn.sigmoid(logits)
        predictions = {
            'prediction': prediction,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        FLAGS.learning_rate, decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate, staircase=True)
    optimizer = tfa.optimizers.LazyAdam(lr_schedule)
    if FLAGS.use_float16:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            optimizer, loss_scale='dynamic')
    optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

    with tf.GradientTape() as tape:
        logits = pbm_reranker(features, training)
        prediction = tf.nn.sigmoid(logits)
        loss = tf.compat.v1.losses.log_loss(labels, prediction)
        if FLAGS.use_float16:
            scaled_loss = optimizer.get_scaled_loss(loss)

    metrics = {
        'auc': tf.compat.v1.metrics.auc(labels, prediction),
        'label/mean': tf.compat.v1.metrics.mean(labels),
        'prediction/mean': tf.compat.v1.metrics.mean(prediction)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    trainable_variables = pbm_reranker.trainable_variables
    if FLAGS.use_float16:
        gradients = tape.gradient(scaled_loss, trainable_variables)
        gradients = optimizer.get_unscaled_gradients(gradients)
    else:
        gradients = tape.gradient(loss, trainable_variables)
    optimize = optimizer.apply_gradients(zip(gradients, trainable_variables))
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    train_op = tf.group([optimize, update_ops])
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
