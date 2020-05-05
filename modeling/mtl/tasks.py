import json
import os

import tensorflow as tf

from absl import flags
from modeling.mtl.expert import Expert
from modeling.mtl.gate import Gate

FLAGS = flags.FLAGS


class Tasks(tf.keras.layers.Layer):
    """Multi-gate Mixture-of-Experts

        ref:
         1) Recommending What Video to Watch Next: A Multitask Ranking System
         2) Multi-Task Deep Neural Networks for Natural Language Understanding
    """

    def __init__(self, config_dir, dropout_rate, gate_dropout_rate):
        super(Tasks, self).__init__()

        mtl_config = self.parse_mtl(config_dir)
        self.task_num = len(mtl_config['tasks'])
        self.experts = []
        for _ in range(self.task_num):
            self.experts.append(Expert(mtl_config['expert'], dropout_rate))

        self.create_gates(self.task_num, gate_dropout_rate)
        self.create_task(mtl_config)

        # shallow tower used for modelling show position bias
        self.st_dense1 = tf.keras.layers.Dense(units=FLAGS.st_filter_size, activation='relu')
        self.st_dense2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=False):
        """inputs contains following two tensor

           input:
              input tensor: shape (B, T, H)
              input sequence mask tensor: shape (B, T)
              item show position tensor: shape (B, T, H')
              tasks labels list: each label's shape (B, T, 1)

           output:
              tasks prediction: dict task -> prediction_tensor (shape: (B, T, 1))
              tasks loss: dict task -> loss
              total loss: scala
        """
        # apply experts on top of shared bottom layer
        # each expert output shape: (B, T, H')
        shared_bottom = inputs[0]
        expert_xs = []
        for expert in self.experts:
            expert_xs.append(expert(shared_bottom, training=training))

        # apply gates on top of shared bottom layer
        # each gate output shape: (B, T, H')
        gate_ws = []
        for gate in self.gates:
            gate_inputs = {'shared_bottom': shared_bottom, 'experts': expert_xs}
            gate_ws.append(gate(gate_inputs, training=training))

        # (B, T, 1)
        sequence_mask = inputs[1]
        weights = tf.expand_dims(tf.cast(sequence_mask, dtype=shared_bottom.dtype), axis=-1)
        # apply tasks on top of gates output layer
        tasks_predictions = {}
        tasks_loss = {}
        total_loss = 0

        # modelling show position bias
        position_bias = self.st_dense1(inputs[2])
        # (B, T, 1)
        position_bias = self.st_dense2(position_bias)

        labels = inputs[3]
        all_predictions = None
        for i, task in enumerate(self.tasks):
            # output shape: (B, T, 1)
            predictions = task['dense'](gate_ws[i])
            # modelling show position bias
            predictions = predictions + position_bias
            predictions = task['activation'](predictions)
            tasks_predictions[task['name']] = predictions
            if training:
                # label shape: (B, T, 1), weights shape: (B, T, 1)
                loss = tf.compat.v1.losses.log_loss(labels=labels[i], predictions=predictions, weights=weights)
                tasks_loss[task['name']] = loss
                total_loss += loss * task['weight']
            if all_predictions is None:
                all_predictions = task['weight'] * predictions
            else:
                all_predictions += task['weight'] * predictions
        tasks_predictions['predictions'] = all_predictions

        return [tasks_predictions, tasks_loss, total_loss]

    def parse_mtl(self, config_dir):
        path = os.path.join(os.path.join(config_dir, 'multi_task'), 'mtl.json')
        with tf.io.gfile.GFile(path) as f:
            mtl_config = json.loads(''.join([line for line in f.readlines()]))
        return mtl_config

    def create_gates(self, expert_num, gate_dropout_rate):
        self.gates = []
        for _ in range(expert_num):
            self.gates.append(Gate(expert_num, gate_dropout_rate))

    def create_task(self, mtl_config):
        self.tasks = []
        for task in mtl_config['tasks']:
            output_size = task['output_size']
            if output_size != 1:
                raise ValueError("Invalid task ouput units: {}".format(output_size))
            self.tasks.append({'dense': tf.keras.layers.Dense(units=output_size),
                               'activation': tf.keras.layers.Activation(activation='sigmoid'),
                               'weight': task['weight'], 'name': task['name']})
