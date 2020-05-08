import json
import collections
import os
import random
import tensorflow as tf

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '', 'where to store train dataset')
flags.DEFINE_integer('train_part_num', 3, 'train file partition num')
flags.DEFINE_string('eval_dir', '', 'where to store eval dataset')
flags.DEFINE_string('config_dir', '', 'feature config dir')
flags.DEFINE_integer('seq_len', 30, 'sequence length')


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
    feature_info = [
        ('user.visited_goods_ids', configs['user.visited_goods_ids']['vocab_size']),
        ('user.visited_shop_ids', configs['user.visited_shop_ids']['vocab_size']),
        ('user.visited_cate_ids', configs['user.visited_cate_ids']['vocab_size']),
        ('user.visited_goods_prices', configs['user.visited_cate_ids']['vocab_size']),
    ]

    for feature, upb in feature_info:
        seq = []
        for _ in range(FLAGS.seq_len):
            seq.append(random.randint(1, upb))
        features[feature] = create_int_feature(seq)


def gen_items_feature(features, configs, training):
    feature_info = [
        ('item.goods_ids', configs['item.goods_ids']['vocab_size']),
        ('item.shop_ids', configs['item.shop_ids']['vocab_size']),
        ('item.cate_ids', configs['item.cate_ids']['vocab_size']),
        ('item.goods_prices', configs['item.cate_ids']['vocab_size']),
        ('item.show_pos', configs['item.show_pos']['vocab_size']),
        ('item.rank_pos', configs['item.rank_pos']['vocab_size']),
    ]

    for feature, upb in feature_info:
        seq = []
        for _ in range(FLAGS.seq_len):
            x = random.randint(1, upb)
            if feature == 'item.show_pos':
                if training:
                    if random.random() < 0.1:  # randomly mask 10% item's show position as unknown
                        x = 0
                else:  # evaluation
                    x = 0
            seq.append(x)
        features[feature] = create_int_feature(seq)


def gen_query_text_features(features, configs):
    features_info = [
        ('user.query_word_ids',
         configs['user.query_word_ids']['vocab_size'],
         configs['user.query_word_ids']['query_len']),
    ]

    for feature_info in features_info:
        feature = feature_info[0]
        upb = feature_info[1]
        total_word_num = feature_info[2]
        seq = []
        for _ in range(total_word_num):
            x = random.randint(1, upb)
            seq.append(x)
        features[feature] = create_int_feature(seq)


def gen_items_text_features(features, configs):
    features_info = [
        ('item.title_word_ids',
         configs['item.title_word_ids']['vocab_size'],
         configs['item.title_word_ids']['title_len']),
        ('item.content_word_ids',
         configs['item.content_word_ids']['vocab_size'],
         configs['item.content_word_ids']['content_len'])
    ]

    for feature_info in features_info:
        feature = feature_info[0]
        upb = feature_info[1]
        total_word_num = feature_info[2] * FLAGS.seq_len
        seq = []
        for _ in range(total_word_num):
            x = random.randint(1, upb)
            seq.append(x)
        features[feature] = create_int_feature(seq)


def gen_context_feature(features, configs):
    feature_info = [
        ('context.hour', configs['context.hour']['vocab_size']),
        ('context.phone', configs['context.phone']['vocab_size'])
    ]
    for feature, upb in feature_info:
        features[feature] = create_int_feature([random.randint(1, upb)])


def gen_click_label(features):
    seq = []
    for _ in range(FLAGS.seq_len):
        seq.append(random.randint(0, 1))
    features["click"] = create_int_feature(seq)


def gen_add_basket_label(features):
    seq = []
    for _ in range(FLAGS.seq_len):
        seq.append(random.randint(0, 1))
    features["add_basket"] = create_int_feature(seq)


def gen_buy_label(features):
    seq = []
    for _ in range(FLAGS.seq_len):
        seq.append(random.randint(0, 1))
    features["buy"] = create_int_feature(seq)


def gen_tfrecord_file(file_name, record_num, configs, training=True):
    writer = tf.io.TFRecordWriter(file_name)
    for _ in range(record_num):
        # feature info
        features = collections.OrderedDict()
        gen_user_feature(features, configs)
        gen_behavior_feature(features, configs)
        gen_query_text_features(features, configs)
        gen_items_feature(features, configs, training)
        gen_items_text_features(features, configs)
        gen_context_feature(features, configs)
        gen_click_label(features)
        gen_add_basket_label(features)
        gen_buy_label(features)
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
    for i in range(FLAGS.train_part_num):
        train_file = os.path.join(FLAGS.train_dir, "part-{:05}".format(i))
        record_num = 10000
        gen_tfrecord_file(train_file, record_num, configs)

    # eval example
    eval_file = os.path.join(FLAGS.eval_dir, "part-00000")
    record_num = 100
    gen_tfrecord_file(eval_file, record_num, configs, training=False)


if __name__ == '__main__':
    app.run(main)
