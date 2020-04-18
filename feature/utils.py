import enum


class FeatureType(enum.Enum):
    categorical = 1
    sequence_categorical = 2
    numerical = 3
    sequence_numerical = 4


class FeatureGroup(enum.Enum):
    user_profile = 1
    user_behavior = 2
    item = 3
    context = 4
    label = 5
