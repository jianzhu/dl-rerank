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
        feature_config = FeatureConfig(config_dir, vocab_dir)

        feature_columns = feature_config.get_feature_columns()

        # check user.gender categorical
        expected_feature = 'user.gender'
        expected_vocab = os.path.join(vocab_dir, 'gender.txt')
        expected_column_type = '.VocabularyFileCategoricalColumn'
        expected_dtype = 'int64'
        self.check_feature_column(expected_feature,
                                  expected_column_type,
                                  expected_dtype,
                                  feature_columns,
                                  expected_vocab=expected_vocab)

        # check user.visited_goods_ids sequence_categorical
        expected_feature = 'user.visited_goods_ids'
        expected_vocab = os.path.join(vocab_dir, 'goods_id.txt')
        expected_column_type = '.SequenceCategoricalColumn'
        expected_dtype = 'int64'
        self.check_feature_column(expected_feature,
                                  expected_column_type,
                                  expected_dtype,
                                  feature_columns,
                                  expected_vocab=expected_vocab)

        # check item.goods_prices sequence_numerical
        expected_feature = 'item.goods_prices'
        expected_column_type = '.SequenceNumericColumn'
        expected_dtype = 'float32'
        self.check_feature_column(expected_feature,
                                  expected_column_type,
                                  expected_dtype,
                                  feature_columns)

        # check label sequence_numerical
        expected_feature = 'label'
        expected_column_type = '.SequenceNumericColumn'
        expected_dtype = 'float32'
        self.check_feature_column(expected_feature,
                                  expected_column_type,
                                  expected_dtype,
                                  feature_columns)

    def test_get_embedding_columns(self):
        root_path = Path(__file__).parent.parent
        config_dir = os.path.join(root_path, 'resources/config')
        vocab_dir = os.path.join(root_path, 'resources/vocab')
        feature_config = FeatureConfig(config_dir, vocab_dir)

        tf.compat.v1.disable_eager_execution()
        embedding_columns = feature_config.get_embedding_columns()

        # check user.gender embedding type
        expected_features = [('user.gender', '.EmbeddingColumn'), ('item.goods_ids', '.SharedEmbeddingColumn')]
        for expected_feature, expected_embedding_type in expected_features:
            self.assertTrue(expected_feature in embedding_columns)
            embedding_column = embedding_columns[expected_feature]
            # assert embedding column type
            self.assertTrue(expected_embedding_type in str(type(embedding_column)))


if __name__ == '__main__':
    unittest.main()
