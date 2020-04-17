from feature_config import FEATURE_CONFIG
from absl import app
from absl import flags
import os

FLAGS = flags.FLAGS

flags.DEFINE_string('vocab_dir', '', 'where to store vocabulary')


def main(_):
    if FLAGS.vocab_dir != '':
        os.makedirs(FLAGS.vocab_dir, exist_ok=True)

    vocab_info = {}
    for _, desc in FEATURE_CONFIG.items():
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
