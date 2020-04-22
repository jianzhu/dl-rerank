import json
import os
import unittest

from utils import shard_info


class TestGetShardInfoMethods(unittest.TestCase):
    def test_get_shard_info_local(self):
        # test local config
        shard_num, shard_id = shard_info.get_shard_info()
        self.assertEqual(1, shard_num)
        self.assertEqual(0, shard_id)

    def test_get_shard_info_chief(self):
        # test chief shard info
        os.environ['TF_CONFIG'] = json.dumps({
            "cluster": {
                "chief": ["localhost:2222"],
                "worker": ["localhost:2223", "localhost:2224", "localhost:2225"],
                "ps": ["localhost:2226", "localhost:2227"]
            },
            "task": {"type": "chief", "index": 0}
        })
        print(os.environ['TF_CONFIG'])
        shard_num, shard_id = shard_info.get_shard_info()
        self.assertEqual(4, shard_num)
        self.assertEqual(0, shard_id)

    def test_get_shard_info_worker(self):
        # test worker shard info
        os.environ['TF_CONFIG'] = json.dumps({
            "cluster": {
                "chief": ["localhost:2222"],
                "worker": ["localhost:2222", "localhost:2222", "localhost:2222"],
                "ps": ["localhost:2222", "localhost:2222"]
            },
            "task": {"type": "worker", "index": 0}
        })
        print(os.environ['TF_CONFIG'])

        shard_num, shard_id = shard_info.get_shard_info()
        self.assertEqual(4, shard_num)
        self.assertEqual(1, shard_id)

    def test_get_shard_info_evaluator(self):
        # test evaluator shard info
        os.environ['TF_CONFIG'] = json.dumps({
            "cluster": {
                "chief": ["localhost:2222"],
                "worker": ["localhost:2222", "localhost:2222", "localhost:2222"],
                "ps": ["localhost:2222", "host5:2222"]
            },
            "task": {"type": "evaluator", "index": 0}
        })
        print(os.environ['TF_CONFIG'])

        shard_num, shard_id = shard_info.get_shard_info()
        self.assertEqual(1, shard_num)
        self.assertEqual(0, shard_id)


if __name__ == '__main__':
    unittest.main()
