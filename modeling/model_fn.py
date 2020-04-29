import tensorflow as tf
import tensorflow_addons as tfa

from absl import flags

from modeling.prm import PRM

FLAGS = flags.FLAGS


def model_fn(features, labels, mode, params):
    feature_config = params['feature_config']
    pbm_reranker = PRM(feature_config)

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.PREDICT:
        outputs = pbm_reranker(features, training)
        prediction = tf.nn.sigmoid(outputs[0])
        predictions = {
            'prediction': prediction,
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
        outputs = pbm_reranker(features, training)
        # (B, T, 1)
        logits = outputs[0]
        # (B, T)
        weights = tf.expand_dims(tf.cast(outputs[1], dtype=logits.dtype), axis=-1)
        prediction = tf.nn.sigmoid(logits)
        loss = tf.compat.v1.losses.log_loss(labels, prediction, weights)
        # add mba l2 reg loss created during forward pass
        #tf.print(pbm_reranker.losses)
        loss += sum(pbm_reranker.losses)
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
