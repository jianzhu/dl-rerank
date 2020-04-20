import os
import collections
import json

import tensorflow as tf
from feature.utils import FeatureType


class FeatureConfig(object):
    """ feature config parser

    config path setting:
        config
           |
           |-- user_profile.json
           |-- user_behavior.json
           |-- items.json
           |-- context.json
           |-- label.json
           |
           |-- embedding.json

    sparse config file format (json):
        {
             "user.gender": {
                 "dim": 2,
                 "default": 0,
                 "type": 'categorical',
                 "vocab": "vocab.gender.txt",
                 "vocab_size": 2
             },

             "user.age_level": {
                 "dim": 2,
                 "default": 0,
                 "type": 'categorical',
                 "vocab": "vocab.age_level.txt",
                 "vocab_size": 8
             },
         }
    """

    def __init__(self, config_dir, vocab_dir):
        self.config_dir = config_dir
        self.vocab_dir = vocab_dir
        self.feature_columns = collections.OrderedDict()
        self.embedding_columns = collections.OrderedDict()

    def get_feature_columns(self):
        if len(self.feature_columns) != 0:
            return self.feature_columns

        # load user profile config
        config_files = ['user_profile.json', 'user_behavior.json', 'items.json', 'context.json', 'label.json']
        for config_file in config_files:
            with tf.io.gfile.GFile(os.path.join(self.config_dir, config_file)) as f:
                config = json.loads(''.join([line for line in f.readlines()]))

            for feature, desc in config.items():
                ftype = FeatureType[desc['type']]
                if ftype == FeatureType.categorical:
                    vocab_file = os.path.join(self.vocab_dir, desc['vocab'])
                    fc = tf.feature_column.categorical_column_with_vocabulary_file(key=feature,
                                                                                   vocabulary_file=vocab_file,
                                                                                   default_value=0,
                                                                                   dtype=tf.int64)
                elif ftype == FeatureType.sequence_categorical:
                    vocab_file = os.path.join(self.vocab_dir, desc['vocab'])
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
                self.feature_columns[feature] = fc
        return self.feature_columns

    def get_embedding_columns(self):
        if len(self.embedding_columns) != 0:
            return self.embedding_columns

        path = os.path.join(self.config_dir, 'embedding.json')
        with tf.io.gfile.GFile(path) as f:
            embedding_configs = json.loads(''.join([line for line in f.readlines()]))

        feature_columns = self.get_feature_columns()
        for config in embedding_configs:
            dim = config['dim']
            features = config['features']
            sub_feature_columns = self.get_sub_feature_columns(feature_columns, features)
            if len(sub_feature_columns) == 0:
                raise ValueError("empty feature list in embedding config")
            elif len(sub_feature_columns) == 1:
                sub_embedding_columns = [tf.feature_column.embedding_column(sub_feature_columns[0], dimension=dim)]
            else:
                sub_embedding_columns = tf.feature_column.shared_embeddings(sub_feature_columns, dimension=dim)

            for feature, embedding_column in zip(features, sub_embedding_columns):
                self.embedding_columns[feature] = embedding_column
        return self.embedding_columns

    @staticmethod
    def get_sub_feature_columns(feature_columns, features):
        sub_feature_columns = []
        for feature in features:
            if feature not in feature_columns:
                raise ValueError('invalid feature in embedding config: {}'.format(feature))
            sub_feature_columns.append(feature_columns.get(feature))
        return sub_feature_columns
