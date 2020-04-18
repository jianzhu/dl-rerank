import os

from absl import app
from absl import flags

from feature.feature_config import FeatureConfig

FLAGS = flags.FLAGS

flags.DEFINE_string('vocab_dir', '', 'where to store vocabulary')
flags.DEFINE_string('fconfig_dir', '', 'feature config dir')


def main(_):
    feature_config = FeatureConfig(FLAGS.fconfig_dir)
    fconfigs = feature_config.get_configs()

    vocab_info = {}
    for _, desc in fconfigs.items():
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
