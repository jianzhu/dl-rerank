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
flags.DEFINE_integer('throttle_secs', 60 * 60, 're-evaluate time past (seconds) after last evaluation')
flags.DEFINE_integer('checkpoint_steps', 1000, 'save checkpoints every this many steps')


def main(_):
    tf.compat.v1.disable_eager_execution()

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
    if shard_id == 0:
        logging.info("begin the final evaluation:")
        metrics = estimator.evaluate(input_fn.eval_input_fn(feature_config))
        print(metrics)
        estimator.export_saved_model(FLAGS.model_path, input_fn.build_serving_fn(feature_config))


if __name__ == '__main__':
    tf.get_logger().setLevel("INFO")
    app.run(main)
