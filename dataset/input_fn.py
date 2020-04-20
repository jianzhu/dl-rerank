import tensorflow as tf

from absl import flags

from dataset.loader import DataLoader

FLAGS = flags.FLAGS


def train_input_fn(feature_config):
    loader = DataLoader(feature_config)
    return lambda: loader.load_data(FLAGS.train_files_dir, FLAGS.batch_size)


def eval_input_fn(feature_config):
    loader = DataLoader(feature_config)
    return lambda: loader.load_data(FLAGS.eval_files_dir, FLAGS.batch_size)


def build_serving_fn(feature_config):
    columns = [x for _, x in feature_config.get_feature_columns().items()]
    parse_spec = tf.feature_column.make_parse_example_spec(columns)
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(parse_spec)
