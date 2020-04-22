import enum


class FeatureType(enum.Enum):
    categorical = 1
    sequence_categorical = 2
    numerical = 3
    sequence_numerical = 4
