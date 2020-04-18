import json
import os
import random
import shutil
import tempfile
import unittest

from unittest.mock import MagicMock
from dataset.loader import DatasetLoader
from feature.feature_config import FeatureConfig
from feature.utils import FeatureType


class TestDatasetLoaderMethods(unittest.TestCase):
    @staticmethod
    def create_vocab(vocab_dir, vocab_file):
        with open(os.path.join(vocab_dir, vocab_file), 'w') as f:
            for x in range(random.randint(1, 10)):
                f.write("{}\n".format(x))

    @staticmethod
    def delete_vocab(vocab_dir):
        # Remove the directory after the test
        shutil.rmtree(vocab_dir)

    def test_get_shard_info_local(self):
        # test local config
        fconfig = FeatureConfig("")
        fconfig.get_configs = MagicMock()
        fconfig.get_configs.return_value = {}

        loader = DatasetLoader(fconfig)
        shard_num, shard_id = loader.get_shard_info()
        self.assertEqual(1, shard_num)
        self.assertEqual(0, shard_id)

    def test_get_shard_info_chief(self):
        # test chief shard info
        os.environ['TF_CONFIG'] = json.dumps({
            "cluster": {
                "chief": ["host0:2222"],
                "worker": ["host1:2222", "host2:2222", "host3:2222"],
                "ps": ["host4:2222", "host5:2222"]
            },
            "task": {"type": "chief", "index": 0}
        })
        fconfig = FeatureConfig("")
        fconfig.get_configs = MagicMock()
        fconfig.get_configs.return_value = {}

        loader = DatasetLoader(fconfig)
        shard_num, shard_id = loader.get_shard_info()
        self.assertEqual(4, shard_num)
        self.assertEqual(0, shard_id)

    def test_get_shard_info_worker(self):
        # test worker shard info
        os.environ['TF_CONFIG'] = json.dumps({
            "cluster": {
                "chief": ["host0:2222"],
                "worker": ["host1:2222", "host2:2222", "host3:2222"],
                "ps": ["host4:2222", "host5:2222"]
            },
            "task": {"type": "worker", "index": 0}
        })
        fconfig = FeatureConfig("")
        fconfig.get_configs = MagicMock()
        fconfig.get_configs.return_value = {}

        loader = DatasetLoader(fconfig)
        shard_num, shard_id = loader.get_shard_info()
        self.assertEqual(4, shard_num)
        self.assertEqual(1, shard_id)

    def test_get_shard_info_evaluator(self):
        # test evaluator shard info
        os.environ['TF_CONFIG'] = json.dumps({
            "cluster": {
                "chief": ["host0:2222"],
                "worker": ["host1:2222", "host2:2222", "host3:2222"],
                "ps": ["host4:2222", "host5:2222"]
            },
            "task": {"type": "evaluator", "index": 0}
        })
        fconfig = FeatureConfig("")
        fconfig.get_configs = MagicMock()
        fconfig.get_configs.return_value = {}

        loader = DatasetLoader(fconfig)
        shard_num, shard_id = loader.get_shard_info()
        self.assertEqual(1, shard_num)
        self.assertEqual(0, shard_id)

    def test_get_column_categorical(self):
        feature = 'user.gender'
        config = {'type': FeatureType.categorical, 'vocab': 'gender.txt'}
        vocab_dir = tempfile.mkdtemp()
        self.create_vocab(vocab_dir, config['vocab'])
        fconfig = FeatureConfig("")
        fconfig.get_configs = MagicMock()
        fconfig.get_configs.return_value = {}

        loader = DatasetLoader(fconfig)
        column = loader.get_column(feature, config, vocab_dir)
        column_config = column.get_config()
        # assert column type
        self.assertTrue('.VocabularyFileCategoricalColumn' in str(type(column)))
        # assert feature name
        self.assertEqual(feature, column_config.get('key'))
        # assert vocabulary file
        vocab_file = os.path.join(vocab_dir, config['vocab'])
        self.assertEqual(vocab_file, column_config.get('vocabulary_file'))
        # assert default value
        self.assertEqual(0, column_config.get('default_value'))
        # assert dtype
        self.assertEqual('int64', column_config.get('dtype'))
        self.delete_vocab(vocab_dir)

    def test_get_column_sequence_categorical(self):
        feature = 'user.visited_goods_ids'
        config = {'type': FeatureType.sequence_categorical, 'vocab': 'goods.txt'}
        vocab_dir = tempfile.mkdtemp()
        self.create_vocab(vocab_dir, config['vocab'])
        fconfig = FeatureConfig("")
        fconfig.get_configs = MagicMock()
        fconfig.get_configs.return_value = {}

        loader = DatasetLoader(fconfig)
        column = loader.get_column(feature, config, vocab_dir)
        column_config = column.get_config()['categorical_column']['config']
        # assert column type
        self.assertTrue('.SequenceCategoricalColumn' in str(type(column)))
        # assert feature name
        self.assertEqual(feature, column_config.get('key'))
        # assert vocabulary file
        vocab_file = os.path.join(vocab_dir, config['vocab'])
        self.assertEqual(vocab_file, column_config.get('vocabulary_file'))
        # assert default value
        self.assertEqual(0, column_config.get('default_value'))
        # assert dtype
        self.assertEqual('int64', column_config.get('dtype'))
        self.delete_vocab(vocab_dir)

    def test_get_column_sequence_numerical(self):
        feature = 'user.visited_goods_price'
        config = {'type': FeatureType.sequence_numerical}
        fconfig = FeatureConfig("")
        fconfig.get_configs = MagicMock()
        fconfig.get_configs.return_value = {}

        loader = DatasetLoader(fconfig)
        column = loader.get_column(feature, config)
        column_config = column.get_config()
        # assert column type
        self.assertTrue('.SequenceNumericColumn' in str(type(column)))
        # assert feature name
        self.assertEqual(feature, column_config.get('key'))
        # assert default value
        self.assertAlmostEqual(0.0, column_config.get('default_value'), delta=1e-5)
        # assert dtype
        self.assertEqual('float32', column_config.get('dtype'))


if __name__ == '__main__':
    unittest.main()
