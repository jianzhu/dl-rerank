import os
import unittest
from pathlib import Path

import tensorflow as tf
from feature.feature_config import FeatureConfig


class TestFeatureConfigMethods(unittest.TestCase):

    def check_feature_column(self,
                             expected_feature,
                             expected_column_type,
                             expected_dtype,
                             feature_columns,
                             expected_vocab=None):
        self.assertTrue(expected_feature in feature_columns)
        column = feature_columns[expected_feature]
        # assert feature column type
        self.assertTrue(expected_column_type in str(type(column)))
        column_config = column.get_config()
        if expected_column_type == '.SequenceCategoricalColumn':
            column_config = column.get_config()['categorical_column']['config']

        # assert feature name
        self.assertEqual(expected_feature, column_config.get('key'))
        # assert dtype
        self.assertEqual(expected_dtype, column_config.get('dtype'))
        # assert vocabulary file
        if expected_vocab is not None:
            self.assertEqual(expected_vocab, column_config.get('vocabulary_file'))

    def test_get_feature_columns(self):
        root_path = Path(__file__).parent.parent
        config_dir = os.path.join(root_path, 'resources/config')
        vocab_dir = os.path.join(root_path, 'resources/vocab')

        tf.compat.v1.disable_eager_execution()
        feature_config = FeatureConfig(config_dir, vocab_dir)
        feature_columns = feature_config.get_feature_columns()

        # check embedding column
        expected_feature = 'user.gender'
        expected_type = '.EmbeddingColumn'
        self.assertTrue(expected_feature in feature_columns)
        self.assertTrue(expected_type in str(type(feature_columns[expected_feature])))

        # check shared embedding column
        expected_feature = 'user.visited_goods_ids'
        expected_type = '.SharedEmbeddingColumn'
        self.assertTrue(expected_feature in feature_columns)
        self.assertTrue(expected_type in str(type(feature_columns[expected_feature])))


if __name__ == '__main__':
    unittest.main()
