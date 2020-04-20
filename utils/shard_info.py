import json
import os


def get_shard_info():
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
