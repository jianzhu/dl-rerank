import os

import tensorflow as tf

from tensorflow import feature_column as fc

from utils import shard_info


class DataLoader(object):
    """load tfrecord format data set


    tf-record dataset path config
      hdfs_dir
         |-- train
         |      |-- part-00000
         |
         |-- eval
         |      |-- part-00000
         |
    """

    def __init__(self, feature_config):
        self.columns = [x for _, x in feature_config.get_feature_columns().items()]

    def load_data(self, file_dir, batch_size=512):
        shard_num, shard_id = shard_info.get_shard_info()
        files = tf.data.Dataset.list_files(os.path.join(file_dir, 'part-*'))
        files = files.shard(shard_num, shard_id)
        print_op = tf.print("shard num: {}, shard_id: {}".format(shard_num, shard_id))
        with tf.control_dependencies([print_op]):
            return files.interleave(lambda tf_file: self.load_file(tf_file, batch_size), num_parallel_calls=8)

    def load_file(self, tf_file, batch_size):
        print_op = tf.print("opening file: ", tf_file)
        with tf.control_dependencies([print_op]):
            dataset = tf.data.TFRecordDataset(tf_file, buffer_size=256*1024*1024)
            dataset = dataset.shuffle(buffer_size=batch_size*10, reshuffle_each_iteration=True)
            parse_spec = fc.make_parse_example_spec(self.columns)
            dataset = dataset.map(map_func=lambda x: self.parse_example(x, parse_spec), num_parallel_calls=8)
            dataset = dataset.batch(batch_size=batch_size)
            dataset = dataset.prefetch(buffer_size=batch_size*10)
            return dataset

    def parse_example(self, serialized, columns):
        features = tf.io.parse_example(serialized=serialized, features=columns)
        labels = (tf.expand_dims(tf.sparse.to_dense(features.pop('click')), axis=-1),
                  tf.expand_dims(tf.sparse.to_dense(features.pop('add_basket')), axis=-1),
                  tf.expand_dims(tf.sparse.to_dense(features.pop('buy')), axis=-1))
        return features, labels
