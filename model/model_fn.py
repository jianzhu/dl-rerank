import tensorflow as tf

from model.pbm_reranker import PBMReRanker


def model_fn(features, labels, mode, params):
    feature_config = params['feature_configs']
    dropout_rate = params['dropout_rate']
    pbm_reranker = PBMReRanker(feature_config, rate=dropout_rate)

    training = (mode == tf.estimator.ModeKeys.TRRAIN)

    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction = tf.nn.sigmoid(pbm_reranker(features, training))
        predictions = {
            'prediction': prediction,
            'auc': tf.compat.v1.metrics.auc(labels, prediction)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.GradientTape() as tape:
        prediction = tf.nn.sigmoid(pbm_reranker(features, training))
        loss = tf.compat.v1.losses.log_loss(labels, prediction)

    metrics = {
        'auc': tf.compat.v1.metrics.auc(labels, prediction),
        'label/mean': tf.compat.v1.metrics.mean(labels),
        'prediction/mean': tf.compat.v1.metrics.mean(prediction)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    trainable_variables = pbm_reranker.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimize = optimizer.apply_gradients(zip(gradients, trainable_variables))
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    train_op = tf.group([optimize, update_ops])
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
