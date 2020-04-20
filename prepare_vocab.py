import json
import os

import tensorflow as tf

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('vocab_dir', '', 'where to store vocabulary')
flags.DEFINE_string('config_dir', '', 'feature config dir')


def main(_):
    configs = {}
    for config_file in tf.io.gfile.listdir(FLAGS.config_dir):
        with tf.io.gfile.GFile(os.path.join(FLAGS.config_dir, config_file)) as f:
            configs.update(json.loads(''.join([line for line in f.readlines()])))

    vocab_info = {}
    for _, desc in configs.items():
        if 'vocab' in desc:
            vocab_size = desc['vocab_size']
            vocab_file = desc['vocab']
            vocab_info[vocab_file] = vocab_size

    for vocab, size in vocab_info.items():
        file = os.path.join(FLAGS.vocab_dir, vocab)
        with open(file, 'w') as f:
            for i in range(size):
                f.write("{}\n".format(i+1))


if __name__ == '__main__':
    app.run(main)
