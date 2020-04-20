import os
import collections
import json

import tensorflow as tf
from feature.utils import FeatureType


class FeatureConfig(object):
    """ feature config parser

    1) config path setting:
        config
           |
           |-- feature_column
           |        |-- user_profile.json
           |        |-- user_behavior.json
           |        |-- items.json
           |        |-- context.json
           |
           |-- embedding_column
           |        |-- embedding.json
           |

    2) feature column config file format (json):
        {
             "user.gender": {
                 "default": 0,
                 "type": 'categorical',
                 "vocab": "vocab.gender.txt",
                 "vocab_size": 2
             },

             "user.age_level": {
                 "default": 0,
                 "type": 'categorical',
                 "vocab": "vocab.age_level.txt",
                 "vocab_size": 8
             },
         }

    3) (share) embedding column config file format (json):
        [

          {
            "dim": 2,
            "features": [
              "user.age_level"
            ]
          },
          {
            "dim": 12,
            "features": [
              "user.visited_goods_ids",
              "item.goods_ids"
            ]
          }
        ]
    """

    def __init__(self, config_dir, vocab_dir):
        self.config_dir = config_dir
        self.vocab_dir = vocab_dir
        self.feature_columns = collections.OrderedDict()

    def get_feature_columns(self):
        if len(self.feature_columns) != 0:
            return self.feature_columns

        # generate feature column info
        self.parse_feature_columns(self.config_dir, self.vocab_dir, self.feature_columns)
        # generate embedding column info
        self.parse_embedding_columns(self.config_dir, self.feature_columns)

        return self.feature_columns

    @staticmethod
    def parse_feature_columns(config_dir, vocab_dir, feature_columns):
        fc_dir = os.path.join(config_dir, 'feature_column')
        # load feature column config
        for config_file in tf.io.gfile.listdir(fc_dir):
            with tf.io.gfile.GFile(os.path.join(fc_dir, config_file)) as f:
                config = json.loads(''.join([line for line in f.readlines()]))

            for feature, desc in config.items():
                ftype = FeatureType[desc['type']]
                if ftype == FeatureType.categorical:
                    vocab_file = os.path.join(vocab_dir, desc['vocab'])
                    fc = tf.feature_column.categorical_column_with_vocabulary_file(key=feature,
                                                                                   vocabulary_file=vocab_file,
                                                                                   default_value=0,
                                                                                   dtype=tf.int64)
                elif ftype == FeatureType.sequence_categorical:
                    vocab_file = os.path.join(vocab_dir, desc['vocab'])
                    fc = tf.feature_column.sequence_categorical_column_with_vocabulary_file(key=feature,
                                                                                            vocabulary_file=vocab_file,
                                                                                            default_value=0,
                                                                                            dtype=tf.int64)
                elif ftype == FeatureType.numerical:
                    fc = tf.feature_column.numeric_column(key=feature, dtype=tf.float32)
                elif ftype == FeatureType.sequence_numerical:
                    fc = tf.feature_column.sequence_numeric_column(key=feature, dtype=tf.float32)
                else:
                    raise ValueError('invalid feature type: {}'.format(ftype))
                feature_columns[feature] = fc

        # add label column
        key = 'label'
        feature_columns[key] = tf.feature_column.sequence_numeric_column(key=key, dtype=tf.int64)
        return feature_columns

    @staticmethod
    def parse_embedding_columns(config_dir, feature_columns):
        path = os.path.join(os.path.join(config_dir, 'embedding_column'), 'embedding.json')
        with tf.io.gfile.GFile(path) as f:
            embedding_configs = json.loads(''.join([line for line in f.readlines()]))

        for config in embedding_configs:
            dim = config['dim']
            features = config['features']
            sub_feature_columns = FeatureConfig.get_sub_feature_columns(feature_columns, features)
            if len(sub_feature_columns) == 0:
                raise ValueError("empty feature list in embedding config")
            elif len(sub_feature_columns) == 1:
                sub_embedding_columns = [tf.feature_column.embedding_column(sub_feature_columns[0], dimension=dim)]
            else:
                sub_embedding_columns = tf.feature_column.shared_embeddings(sub_feature_columns, dimension=dim)

            for feature, embedding_column in zip(features, sub_embedding_columns):
                feature_columns[feature] = embedding_column

    @staticmethod
    def get_sub_feature_columns(feature_columns, features):
        sub_feature_columns = []
        for feature in features:
            if feature not in feature_columns:
                raise ValueError('invalid feature in embedding config: {}'.format(feature))
            sub_feature_columns.append(feature_columns.get(feature))
        return sub_feature_columns
