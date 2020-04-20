import tensorflow as tf

from absl import app
from absl import flags

from dataset.loader import DataLoader
from feature.feature_config import FeatureConfig
from model.pbm_reranker import PBMReRanker

FLAGS = flags.FLAGS

# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
flags.DEFINE_string('config_dir', None, 'feature config directory')
flags.DEFINE_string('vocab_dir', None, 'feature vocab directory')
flags.DEFINE_string('train_files_dir', None, 'train data set file directory')
flags.DEFINE_string('eval_files_dir', None, 'eval data set file directory')
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_float('dropout_rate', 0.3, 'dropout rate')


def main(argv):
    # init feature config
    feature_config = FeatureConfig(config_dir=FLAGS.config_dir, vocab_dir=FLAGS.vocab_dir)
    tf.compat.v1.enable_eager_execution()

    # reranker model
    reranker = PBMReRanker(feature_config, rate=FLAGS.dropout_rate)

    # loss and optimizer
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = reranker(x)
            loss = loss_fn(y, logits)
            gradients = tape.gradient(loss, reranker.trainable_weights)
        optimizer.apply_gradients(zip(gradients, reranker.trainable_weights))
        return loss

    # load train & valid dataset
    data_loader = DataLoader(feature_config)
    train_dataset = data_loader.load_data(file_dir=FLAGS.train_files_dir, batch_size=FLAGS.batch_size)
    eval_dataset = data_loader.load_data(file_dir=FLAGS.eval_files_dir, batch_size=FLAGS.batch_size)

    for step, (x, y) in enumerate(train_dataset):
        print(step)
        print(x)
        print(y)


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    app.run(main)
