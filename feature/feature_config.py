import os
import collections
import json

import tensorflow as tf
from feature.utils import FeatureType


class FeatureConfig(object):
    """ feature config parser

    config path setting:
        feature_config
           |
           |-- user_profile.json
           |-- user_behavior.json
           |-- items.json
           |-- context.json
           |-- label.json

    config file format (json):
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
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.configs = collections.OrderedDict()

    def parse_config(self, config_file):
        with tf.io.gfile.GFile(os.path.join(self.config_dir, config_file)) as f:
            config = json.loads(''.join([line for line in f.readlines()]))

        for _, desc in config.items():
            desc['type'] = FeatureType[desc['type']]
        return config

    def get_configs(self):
        if len(self.configs) != 0:
            return self.configs

        # load user profile config
        config_files = ['user_profile.json',
                        'user_behavior.json',
                        'items.json',
                        'context.json',
                        'label.json']
        for config_file in config_files:
            config = self.parse_config(config_file)
            self.configs.update(config)
        return self.configs
