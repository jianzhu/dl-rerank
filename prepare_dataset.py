import collections
import os
import random
import tensorflow as tf

from absl import app
from absl import flags

from feature_config import FEATURE_CONFIG

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', '', 'where to store dataset')


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


# define feature generate help function
def gen_user_feature(features):
    # 1: male, 2: female
    upb = FEATURE_CONFIG['user.gender']['vocab_size']
    gender = [random.randint(1, upb)]
    features['user.gender'] = create_int_feature(gender)
    # 1: < 10, 2: < 20, 3: < 30, ..., 7: < 70, 8: other
    upb = FEATURE_CONFIG['user.age_level']['vocab_size']
    age_level = [random.randint(1, upb)]
    features['user.age_level'] = create_int_feature(age_level)


def gen_behavior_feature(features):
    # visited goods id & shop id & cate id list
    seq_len = random.randint(1, 4)
    feature_info = [
        ('user.visited_goods_ids', FEATURE_CONFIG['user.visited_goods_ids']['vocab_size']),
        ('user.visited_shop_ids', FEATURE_CONFIG['user.visited_shop_ids']['vocab_size']),
        ('user.visited_cate_ids', FEATURE_CONFIG['user.visited_cate_ids']['vocab_size']),
        ('user.visited_goods_price', None),
        ('user.segment', None)
    ]

    for feature, upb in feature_info:
        seq = []
        for _ in range(seq_len):
            if feature == 'user.visited_goods_price':
                seq.append(random.random() * 10)
            elif feature == 'user.segment':
                seq.append(1)
            else:
                seq.append(random.randint(1, upb))
        if feature == 'user.visited_goods_price':
            features[feature] = create_float_feature(seq)
        else:
            features[feature] = create_int_feature(seq)


def gen_ads_feature(features, seq_len=5):
    feature_info = [
        ('ads.goods_ids', FEATURE_CONFIG['ads.goods_ids']['vocab_size']),
        ('ads.shop_ids', FEATURE_CONFIG['ads.shop_ids']['vocab_size']),
        ('ads.cate_ids', FEATURE_CONFIG['ads.cate_ids']['vocab_size']),
        ('ads.goods_prices', None),
        ('ads.segment', None)
    ]

    for feature, upb in feature_info:
        seq = []
        for _ in range(seq_len):
            if feature == 'ads.goods_prices':
                seq.append(random.random() * 10)
            elif feature == 'ads.segment':
                seq.append(2)
            else:
                seq.append(random.randint(1, upb))
        if feature == 'ads.goods_prices':
            features[feature] = create_float_feature(seq)
        else:
            features[feature] = create_int_feature(seq)


def gen_context_feature(features):
    feature_info = [
        ('context.hour', FEATURE_CONFIG['context.hour']['vocab_size']),
        ('context.phone', FEATURE_CONFIG['context.phone']['vocab_size'])
    ]
    for feature, upb in feature_info:
        features[feature] = create_int_feature([random.randint(1, upb)])


def gen_labels(features, seq_len=5):
    seq = []
    for _ in range(seq_len):
        seq.append(float(random.randint(0, 1)))
    features["labels"] = create_float_feature(seq)


def gen_tfrecord_file(file_name, record_num):
    writer = tf.io.TFRecordWriter(file_name)
    for _ in range(record_num):
        # feature info
        features = collections.OrderedDict()
        gen_user_feature(features)
        gen_behavior_feature(features)
        gen_ads_feature(features)
        gen_context_feature(features)
        gen_labels(features)
        # write example
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
    writer.close()


def main(_):
    if FLAGS.dataset_dir != '':
        os.makedirs(FLAGS.dataset_dir, exist_ok=True)

    # train example
    train_file = os.path.join(FLAGS.dataset_dir, "train.part-00000")
    record_num = 10000
    gen_tfrecord_file(train_file, record_num)

    # eval example
    eval_file = os.path.join(FLAGS.dataset_dir, "eval.part-00000")
    record_num = 100
    gen_tfrecord_file(eval_file, record_num)


if __name__ == '__main__':
    app.run(main)
