import numpy as np
import enum


class FeatureType(enum.Enum):
    categorical = 1
    sequence_categorical = 2
    numerical = 3
    sequence_numerical = 4
    label = 5


FEATURE_CONFIG = {
    # user profile features
    "user.gender": {
        "dim": 2,
        "default": 0,
        "type": FeatureType.categorical,
        "vocab": "vocab.gender.txt",
        "vocab_size": 2
    },

    "user.age_level": {
        "dim": 2,
        "default": 0,
        "type": FeatureType.categorical,
        "vocab": "vocab.age_level.txt",
        "vocab_size": 8
    },

    # user behavior features
    "user.visited_goods_ids": {
        "dim": 12,
        "default": 0,
        "type": FeatureType.sequence_categorical,
        "vocab": "vocab.goods_id.txt",
        "vocab_size": np.int64(1e6)
    },

    "user.visited_shop_ids": {
        "dim": 6,
        "default": 0,
        "type": FeatureType.sequence_categorical,
        "vocab": "vocab.shop_id.txt",
        "vocab_size": np.int64(1e4)
    },

    "user.visited_cate_ids": {
        "dim": 6,
        "default": 0,
        "type": FeatureType.sequence_categorical,
        "vocab": "vocab.cate_id.txt",
        "vocab_size": np.int64(1e3)
    },

    "user.visited_goods_price": {
        "dim": 1,
        "default": 0,
        "type": FeatureType.sequence_numerical
    },

    # ads features
    "ads.goods_ids": {
        "dim": 12,
        "default": 0,
        "type": FeatureType.sequence_categorical,
        "vocab": "vocab.goods_id.txt",
        "vocab_size": np.int64(1e6)
    },

    "ads.shop_ids": {
        "dim": 6,
        "default": 0,
        "type": FeatureType.sequence_categorical,
        "vocab": "vocab.shop_id.txt",
        "vocab_size": np.int64(1e4)
    },

    "ads.cate_ids": {
        "dim": 6,
        "default": 0,
        "type": FeatureType.sequence_categorical,
        "vocab": "vocab.cate_id.txt",
        "vocab_size": np.int64(1e3)
    },

    "ads.goods_prices": {
        "dim": 1,
        "default": 0,
        "type": FeatureType.sequence_numerical
    },

    # context features
    "context.hour": {
        "dim": 2,
        "default": 0,
        "type": FeatureType.categorical,
        "vocab": "vocab.hour.txt",
        "vocab_size": 24
    },

    "context.phone": {
        "dim": 6,
        "default": 0,
        "type": FeatureType.categorical,
        "vocab": "vocab.phone.txt",
        "vocab_size": np.int64(1e3)
    },

    # labels
    "labels": {
        "dim": 1,
        "default": 0,
        "type": FeatureType.label
    }
}
