import os
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging

from dataset import input_fn
from feature import feature_config as fc
from model import model_fn
from utils import shard_info

FLAGS = flags.FLAGS

flags.DEFINE_string('config_dir', None, 'feature config directory')
flags.DEFINE_string('vocab_dir', None, 'feature vocab directory')
flags.DEFINE_string('train_files_dir', None, 'train data set file directory')
flags.DEFINE_string('eval_files_dir', None, 'eval data set file directory')
flags.DEFINE_string('model_path', 'pbm_reranker', 'model path')
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_float('dropout_rate', 0.3, 'dropout rate')
flags.DEFINE_integer('train_max_steps', 10000, 'maximum train steps')
flags.DEFINE_integer('eval_steps', -1, 'eval steps')
flags.DEFINE_integer('throttle_secs', 60, 're-evaluate time past (seconds) after last evaluation')
flags.DEFINE_integer('checkpoint_steps', 2000, 'save checkpoints every this many steps')
flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate for adam')
flags.DEFINE_integer('decay_steps', 10000, 'decay steps')
flags.DEFINE_float('decay_rate', 0.96, 'decay rate')
flags.DEFINE_float('l2_reg_w', 1e-4, 'mini-batch aware regularization weight')

# performance flags
flags.DEFINE_bool('enable_xla', True, 'enable xla')
flags.DEFINE_bool('use_float16', True, 'use float16 mixed_precision')


def performance_optimize():
    # enable xla
    if FLAGS.enable_xla:
        tf.config.optimizer.set_jit(True)

    # enable layer fp16 mixed precision computation
    if FLAGS.use_float16:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16', loss_scale='dynamic')
        tf.keras.mixed_precision.experimental.set_policy(policy)


def main(_):
    tf.compat.v1.disable_eager_execution()

    performance_optimize()

    feature_config = fc.FeatureConfig(
        config_dir=FLAGS.config_dir, vocab_dir=FLAGS.vocab_dir)

    train_spec = tf.estimator.TrainSpec(
        input_fn.train_input_fn(feature_config), max_steps=FLAGS.train_max_steps)

    eval_spec = tf.estimator.EvalSpec(
        input_fn.eval_input_fn(feature_config),
        steps=None if FLAGS.eval_steps < 0 else FLAGS.eval_steps, throttle_secs=FLAGS.throttle_secs)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_path, save_checkpoints_steps=FLAGS.checkpoint_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn.model_fn, config=run_config, params={
            'feature_config': feature_config,
            'dropout_rate': FLAGS.dropout_rate
        })

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    _, shard_id = shard_info.get_shard_info()
    if 'TF_CONFIG' not in os.environ or shard_id == 0:
        logging.info("begin the final evaluation:")
        metrics = estimator.evaluate(input_fn.eval_input_fn(feature_config))
        print(metrics)
        estimator.export_saved_model(os.path.join(FLAGS.model_path, 'final'),
                                     input_fn.build_serving_fn(feature_config))


if __name__ == '__main__':
    tf.get_logger().setLevel("INFO")
    app.run(main)
