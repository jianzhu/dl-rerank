import tensorflow as tf
import tensorflow_addons as tfa

from absl import flags

from modeling.prm import PRM

FLAGS = flags.FLAGS


def model_fn(features, labels, mode, params):
    feature_config = params['feature_config']
    dropout_rate = params['dropout_rate']
    layer_num = params['layer_num']
    head_num = params['head_num']
    hidden_size = params['hidden_size']
    filter_size = params['filter_size']
    pbm_reranker = PRM(feature_config,
                       layer_num=layer_num,
                       head_num=head_num,
                       hidden_size=hidden_size,
                       filter_size=filter_size,
                       dropout_rate=dropout_rate)

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
        #print_op = tf.print(pbm_reranker.losses)
        #with tf.control_dependencies([print_op]):
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
    global_step = tf.compat.v1.train.get_or_create_global_step()
    update_global_step = tf.compat.v1.assign(global_step, global_step + 1, name='update_global_step')
    train_op = tf.group([optimize, update_global_step])
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
