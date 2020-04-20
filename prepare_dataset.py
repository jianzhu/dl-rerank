import json
import collections
import os
import random
import tensorflow as tf

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '', 'where to store train dataset')
flags.DEFINE_string('eval_dir', '', 'where to store eval dataset')
flags.DEFINE_string('config_dir', '', 'feature config dir')


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


# define feature generate help function
def gen_user_feature(features, configs):
    # 1: male, 2: female
    upb = configs['user.gender']['vocab_size']
    gender = [random.randint(1, upb)]
    features['user.gender'] = create_int_feature(gender)
    # 1: < 10, 2: < 20, 3: < 30, ..., 7: < 70, 8: other
    upb = configs['user.age_level']['vocab_size']
    age_level = [random.randint(1, upb)]
    features['user.age_level'] = create_int_feature(age_level)


def gen_behavior_feature(features, configs):
    # visited goods id & shop id & cate id list
    seq_len = random.randint(1, 4)
    feature_info = [
        ('user.visited_goods_ids', configs['user.visited_goods_ids']['vocab_size']),
        ('user.visited_shop_ids', configs['user.visited_shop_ids']['vocab_size']),
        ('user.visited_cate_ids', configs['user.visited_cate_ids']['vocab_size']),
        ('user.visited_goods_price', None),
    ]

    for feature, upb in feature_info:
        seq = []
        for i in range(seq_len):
            if feature == 'user.visited_goods_price':
                seq.append(random.random() * 10)
            else:
                rnd = random.randint(1, upb)
                seq.append(rnd)
        if feature == 'user.visited_goods_price':
            features[feature] = create_float_feature(seq)
        else:
            features[feature] = create_int_feature(seq)


def gen_ads_feature(features, configs, seq_len=5):
    feature_info = [
        ('item.goods_ids', configs['item.goods_ids']['vocab_size']),
        ('item.shop_ids', configs['item.shop_ids']['vocab_size']),
        ('item.cate_ids', configs['item.cate_ids']['vocab_size']),
        ('item.goods_prices', None),
    ]

    for feature, upb in feature_info:
        seq = []
        for _ in range(seq_len):
            if feature == 'item.goods_prices':
                seq.append(random.random() * 10)
            else:
                seq.append(random.randint(1, upb))
        if feature == 'item.goods_prices':
            features[feature] = create_float_feature(seq)
        else:
            features[feature] = create_int_feature(seq)


def gen_context_feature(features, configs):
    feature_info = [
        ('context.hour', configs['context.hour']['vocab_size']),
        ('context.phone', configs['context.phone']['vocab_size'])
    ]
    for feature, upb in feature_info:
        features[feature] = create_int_feature([random.randint(1, upb)])


def gen_label(features, seq_len=5):
    seq = []
    for _ in range(seq_len):
        seq.append(random.randint(0, 1))
    features["label"] = create_int_feature(seq)


def gen_tfrecord_file(file_name, record_num, configs):
    writer = tf.io.TFRecordWriter(file_name)
    for _ in range(record_num):
        # feature info
        features = collections.OrderedDict()
        gen_user_feature(features, configs)
        gen_behavior_feature(features, configs)
        gen_ads_feature(features, configs)
        gen_context_feature(features, configs)
        gen_label(features)
        # write example
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
    writer.close()


def main(_):
    configs = {}
    for config_file in tf.io.gfile.listdir(FLAGS.config_dir):
        with tf.io.gfile.GFile(os.path.join(FLAGS.config_dir, config_file)) as f:
            configs.update(json.loads(''.join([line for line in f.readlines()])))

    # train example
    train_file = os.path.join(FLAGS.train_dir, "part-00000")
    record_num = 10000
    gen_tfrecord_file(train_file, record_num, configs)

    # eval example
    eval_file = os.path.join(FLAGS.eval_dir, "part-00000")
    record_num = 100
    gen_tfrecord_file(eval_file, record_num, configs)


if __name__ == '__main__':
    app.run(main)
