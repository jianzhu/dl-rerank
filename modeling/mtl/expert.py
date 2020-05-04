import tensorflow as tf


class Expert(tf.keras.layers.Layer):
    """Expert layer on top of Shared Bottom Layer

       ref: Recommending What Video to Watch Next: A Multitask Ranking System

       expert config format:
          [
            {
              "layer_units": 64,
              "activation": "relu"
            },
            {
              "layer_units": 32,
              "activation": "relu"
            }
          ]
    """

    def __init__(self, expert_config, dropout_rate):
        super(Expert, self).__init__()

        self.layers = []
        for layer in expert_config:
            bn = tf.keras.layers.BatchNormalization(epsilon=1e-6)
            drop = tf.keras.layers.Dropout(rate=dropout_rate)
            dense = tf.keras.layers.Dense(units=layer['layer_units'], activation=layer['activation'])
            self.layers.append((bn, drop, dense))

    def call(self, inputs, training=False):
        """Expert representation transformation layer

           input:
               input tensor: shape (B, T, H)
           output:
               output tensor: shape (B, T, H')
        """
        x = inputs
        for bn, drop, dense in self.layers:
            x = bn(x, training=training)
            x = drop(x, training=training)
            x = dense(x)
        return x
