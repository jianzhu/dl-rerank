import tensorflow as tf
import tensorflow_addons as tfa

from absl import flags

from modeling.prm import PRM

FLAGS = flags.FLAGS


def model_fn(features, labels, mode, params):
    feature_config = params['feature_config']
    pbm_reranker = PRM(feature_config)

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    inputs = [features, labels]
    if mode == tf.estimator.ModeKeys.PREDICT:
        tasks_predictions, tasks_loss, total_loss = pbm_reranker(inputs, training)
        predictions = {
            'prediction': tasks_predictions['predictions'],
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        FLAGS.learning_rate, decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate, staircase=True)
    optimizer = tfa.optimizers.LazyAdam(lr_schedule)
    optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
    if FLAGS.use_float16:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            optimizer, loss_scale='dynamic')

    with tf.GradientTape() as tape:
        tasks_predictions, tasks_loss, total_loss = pbm_reranker(inputs, training)
        loss = total_loss
        # add mba l2 reg loss created during forward pass
        loss += sum(pbm_reranker.losses)
        if FLAGS.use_float16:
            scaled_loss = optimizer.get_scaled_loss(loss)

    metrics = {
        'click_auc': tf.compat.v1.metrics.auc(labels[0], tasks_predictions['click']),
        'click_label/mean': tf.compat.v1.metrics.mean(labels[0]),
        'click_prediction/mean': tf.compat.v1.metrics.mean(tasks_predictions['click']),
        'add_basket_auc': tf.compat.v1.metrics.auc(labels[1], tasks_predictions['add_basket']),
        'add_basket_label/mean': tf.compat.v1.metrics.mean(labels[1]),
        'add_basket_prediction/mean': tf.compat.v1.metrics.mean(tasks_predictions['add_basket']),
        'buy_auc': tf.compat.v1.metrics.auc(labels[2], tasks_predictions['buy']),
        'buy_label/mean': tf.compat.v1.metrics.mean(labels[2]),
        'buy_prediction/mean': tf.compat.v1.metrics.mean(tasks_predictions['buy']),
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
