import os
import unittest
from pathlib import Path

from feature.utils import FeatureType
from feature.feature_config import FeatureConfig


class TestFeatureConfigMethods(unittest.TestCase):

    def test_parse_config_user_profile(self):
        root_path = Path(__file__).parent.parent
        config_dir = os.path.join(root_path, 'resources/feature_config')
        feature_config = FeatureConfig(config_dir)

        config_file = 'user_profile.json'
        config = feature_config.parse_config(config_file)

        # assert gender
        self.assertTrue('user.gender' in config)
        gender = config['user.gender']
        self.assertEqual(2, gender['dim'])
        self.assertEqual(0, gender['default'])
        self.assertEqual(FeatureType.categorical, gender['type'])
        self.assertEqual('gender.txt', gender['vocab'])
        self.assertEqual(2, gender['vocab_size'])

    def test_parse_config_user_behavior(self):
        root_path = Path(__file__).parent.parent
        config_dir = os.path.join(root_path, 'resources/feature_config')
        feature_config = FeatureConfig(config_dir)
        config_file = 'user_behavior.json'
        config = feature_config.parse_config(config_file)

        # assert visited_goods_ids
        self.assertTrue('user.visited_goods_ids' in config)
        visited_gids = config['user.visited_goods_ids']
        self.assertEqual(12, visited_gids['dim'])
        self.assertEqual(0, visited_gids['default'])
        self.assertEqual(FeatureType.sequence_categorical, visited_gids['type'])
        self.assertEqual('goods_id.txt', visited_gids['vocab'])
        self.assertEqual(100, visited_gids['vocab_size'])

        # assert visited_goods_price
        self.assertTrue('user.visited_goods_price' in config)
        visited_gids = config['user.visited_goods_price']
        self.assertEqual(1, visited_gids['dim'])
        self.assertEqual(0, visited_gids['default'])
        self.assertEqual(FeatureType.sequence_numerical, visited_gids['type'])

    def test_parse_config_label(self):
        root_path = Path(__file__).parent.parent
        config_dir = os.path.join(root_path, 'resources/feature_config')
        feature_config = FeatureConfig(config_dir)
        config_file = 'label.json'
        config = feature_config.parse_config(config_file)

        # assert label
        self.assertTrue('label' in config)
        label = config['label']
        self.assertEqual(1, label['dim'])
        self.assertEqual(0, label['default'])
        self.assertEqual(FeatureType.sequence_numerical, label['type'])

    def test_get_configs(self):
        root_path = Path(__file__).parent.parent
        config_dir = os.path.join(root_path, 'resources/feature_config')
        feature_config = FeatureConfig(config_dir)

        configs = feature_config.get_configs()

        # assert keys
        self.assertTrue('user.gender' in configs)
        self.assertTrue('user.age_level' in configs)
        self.assertTrue('user.visited_goods_ids' in configs)
        self.assertTrue('item.goods_ids' in configs)
        self.assertTrue('context.phone' in configs)
        self.assertTrue('label' in configs)


if __name__ == '__main__':
    unittest.main()
