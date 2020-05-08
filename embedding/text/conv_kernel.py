import tensorflow as tf


class ItemsTextConv(tf.keras.layers.Layer):
    """ Text Representation with Convolutional Neural Network

        ref:
        1) Convolutional Neural Networks for Sentence Classification
        2) Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks
    """

    def __init__(self, filters, kernel_sizes, seq_len):
        super(ItemsTextConv, self).__init__()

        self.filters = filters
        self.kernel_sizes = [int(x) for x in kernel_sizes.split(',')]
        self.conv_pools = [tf.keras.models.Sequential(layers=[
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=self.filters,
                                                                   kernel_size=kernel_size,
                                                                   activation='relu',
                                                                   padding='same')),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool1D(pool_size=seq_len)),
        ]) for kernel_size in self.kernel_sizes]

    def call(self, inputs, training=False):
        """ convolution transformation
           inputs shape: (B, T, W, H)
              # B - batch size
              # T - item num
              # W - item word id num
              # H - word id embedding size

           outputs shape: (B, T, filters x len(kernel_sizes))
        """
        outputs = [conv_pool(inputs) for conv_pool in self.conv_pools]
        # shape: (B, T, 1, filers x len(kernel_sizes))
        outputs = tf.concat(outputs, axis=-1)
        # shape: (B, T, filers x len(kernel_sizes))
        outputs = tf.squeeze(outputs, axis=[-2])

        # (B, T, filters x len(kernel_sizes))
        return outputs


class QueryTextConv(tf.keras.layers.Layer):
    """ Text Representation with Convolutional Neural Network

        ref:
        1) Convolutional Neural Networks for Sentence Classification
        2) Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks
    """

    def __init__(self, filters, kernel_sizes, seq_len):
        super(QueryTextConv, self).__init__()

        self.filters = filters
        self.kernel_sizes = [int(x) for x in kernel_sizes.split(',')]
        self.conv_pools = [tf.keras.models.Sequential(layers=[
            tf.keras.layers.Conv1D(filters=self.filters,
                                   kernel_size=kernel_size,
                                   activation='relu',
                                   padding='same'),
            tf.keras.layers.MaxPool1D(pool_size=seq_len),
        ]) for kernel_size in self.kernel_sizes]

    def call(self, inputs, training=False):
        """ convolution transformation
           inputs shape: (B, W, H)
              # B - batch size
              # W - query word id num
              # H - word id embedding size

           outputs shape: (B, 1, filters x len(kernel_sizes))
        """
        outputs = [conv_pool(inputs) for conv_pool in self.conv_pools]
        # shape: (B, 1, filers x len(kernel_sizes))
        return tf.concat(outputs, axis=-1)
