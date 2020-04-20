import json
import os

import tensorflow as tf
from tensorflow import feature_column as fc

from feature.utils import FeatureType


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
         |-- vocab
         |      |-- age.txt
    """

    def __init__(self, features_config):
        self.configs = features_config.get_configs()

    def load_data(self, file_dir, vocab_dir, batch_size=512):
        shard_num, shard_id = self.get_shard_info()
        files = tf.data.Dataset.list_files(os.path.join(file_dir, 'part-*'))
        files = files.shard(shard_num, shard_id)
        return files.flat_map(lambda tf_file: self.load_file(tf_file, vocab_dir, batch_size))

    def get_shard_info(self):
        tf_config = os.environ.get('TF_CONFIG', None)
        if tf_config is None:
            return 1, 0

        config = json.loads(tf_config)
        worker_num = len(config['cluster']['worker'])
        chief_num = len(config['cluster']['chief'])
        shard_num = worker_num + chief_num
        if config['task']['type'] == 'chief':
            shard_id = 0
        elif config['task']['type'] == 'worker':
            shard_id = config['task']['index'] + 1
        elif config['task']['type'] == 'evaluator':
            shard_num = 1
            shard_id = 0
        else:
            raise ValueError('invalid get_shard_info apply logic')
        return shard_num, shard_id

    def load_file(self, tf_file, vocab_dir, batch_size):
        columns = []
        for feature, config in self.configs.items():
            columns.append(self.get_column(feature, config, vocab_dir))

        dataset = tf.data.TFRecordDataset(tf_file, buffer_size=256*1024*1024)
        #dataset = dataset.shuffle(buffer_size=batch_size*10, reshuffle_each_iteration=True)
        parse_spec = fc.make_parse_example_spec(columns)
        dataset = dataset.map(map_func=lambda x: self.parse_example(x, parse_spec), num_parallel_calls=8)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=batch_size*10)
        return dataset

    def get_column(self, feature, config, vocab_dir=''):
        feature_type = config['type']
        if feature_type == FeatureType.categorical:
            vocab_file = os.path.join(vocab_dir, config['vocab'])
            ft = fc.categorical_column_with_vocabulary_file(key=feature,
                                                            vocabulary_file=vocab_file,
                                                            default_value=0,
                                                            dtype=tf.int64)
        elif feature_type == FeatureType.numerical:
            ft = fc.numeric_column(key=feature, dtype=tf.float32)
        elif feature_type == FeatureType.sequence_categorical:
            vocab_file = os.path.join(vocab_dir, config['vocab'])
            ft = fc.sequence_categorical_column_with_vocabulary_file(key=feature,
                                                                     vocabulary_file=vocab_file,
                                                                     default_value=0,
                                                                     dtype=tf.int64)
            ft = fc.embedding_column(ft, dimension=3)
        elif feature_type == FeatureType.sequence_numerical:
            ft = fc.sequence_numeric_column(key=feature, dtype=tf.float32)
        else:
            raise ValueError("invalid feature type {}".format(feature_type))
        return ft

    def parse_example(self, serialized, columns):
        features = tf.io.parse_example(serialized=serialized, features=columns)
        label = tf.expand_dims(tf.sparse.to_dense(features.pop('label')), axis=-1)
        return features, label


from feature.feature_config import FeatureConfig
from pathlib import Path

root_path = Path(__file__).parent.parent
config_dir = os.path.join(root_path, 'resources/config')
feature_config = FeatureConfig(config_dir)
data_loader = DataLoader(feature_config)

file_dir = os.path.join(root_path, 'resources/train')
vocab_dir = os.path.join(root_path, 'resources/vocab')
dataset = data_loader.load_data(file_dir, vocab_dir, 2)

features, label = next(iter(dataset))

from tensorflow import keras
print(keras.layers.DenseFeatures())

print('user.visited_shop_ids')
print(features['user.visited_shop_ids'])

print('user.visited_goods_ids')
print(tf.sparse.to_dense(features['user.visited_goods_ids']))

print('user.visited_cate_ids')
print(tf.sparse.to_dense(features['user.visited_cate_ids']))

print('item.shop_ids')
print(tf.sparse.to_dense(features['item.shop_ids']))

print('item.goods_ids')
print(tf.sparse.to_dense(features['item.goods_ids']))

print('item.cate_ids')
print(tf.sparse.to_dense(features['item.cate_ids']))
