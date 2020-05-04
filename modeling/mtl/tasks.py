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

    def __init__(self, config_dir, dropout_rate):
        super(Tasks, self).__init__()

        mtl_config = self.parse_mtl(config_dir)
        self.task_num = len(mtl_config['tasks'])
        self.experts = []
        for _ in range(self.task_num):
            self.experts.append(Expert(mtl_config['expert'], dropout_rate))

        self.create_gates(self.task_num)
        self.create_task(mtl_config)

    def call(self, inputs, training=False):
        """inputs contains following two tensor

           input:
              input tensor: shape (B, T, H)
              input sequence mask tensor: shape (B, T)
              tasks labels list: each label's shape (B, T, 1)

           output:
              output tensor: shape (B, T, H)
        """
        # apply experts on top of shared bottom layer
        # each expert output shape: (B, T, H')
        expert_xs = []
        for expert in self.experts:
            expert_xs.append(expert(inputs[0], training=training))

        # apply gates on top of shared bottom layer
        # each gate output shape: (B, T, H')
        gate_ws = []
        for gate in self.gates:
            gate_inputs = {'shared_bottom': inputs[0], 'experts': expert_xs}
            gate_ws.append(gate(gate_inputs, training=training))

        weights = tf.expand_dims(tf.cast(inputs[1], dtype=inputs[0].dtype), axis=-1)
        # apply tasks on top of gates output layer
        tasks_output = []
        tasks_loss = []
        total_loss = 0
        for gate_x, task, label in zip(gate_ws, self.tasks, inputs[2]):
            # output shape: (B, T, 1)
            output = task['dense'](gate_x)
            tasks_output.append({'name': task['name'], 'output': output})
            # label shape: (B, T, 1), mask shape: (B, T)
            if task['type'] == 'classification' and task['unit'] == 1:
                loss = tf.compat.v1.losses.log_loss(label, output, weights)
            elif task['type'] == 'regression':
                loss = tf.compat.v1.losses.mean_squared_error(label, output, weights)
            else:
                raise ValueError("unsupported task loss type")
            total_loss += loss * task['weight']
            tasks_loss.append({'name': task['name'], 'loss': loss})
        return tasks_output, tasks_loss, total_loss

    def parse_mtl(self, config_dir):
        path = os.path.join(os.path.join(config_dir, 'multi_task'), 'mtl.json')
        with tf.io.gfile.GFile(path) as f:
            mtl_config = json.loads(''.join([line for line in f.readlines()]))
        return mtl_config

    def create_gates(self, expert_num):
        self.gates = []
        for _ in range(expert_num):
            self.gates.append(Gate(expert_num, FLAGS.gate_dropout))

    def create_task(self, mtl_config):
        self.tasks = []
        for task in mtl_config['tasks']:
            type = task['type']
            units = task['units']
            if units < 1:
                raise ValueError("Invalid task ouput units: {}".format(units))

            if type == 'classification':
                if units == 1:
                    dense = tf.keras.layers.Dense(units=units, activation='sigmoid')
                else:
                    dense = tf.keras.layers.Dense(units=units, activation='softmax')
            elif type == 'regression':
                if units != 1:
                    raise ValueError("Regression task output units > 1: {}".format(units))
                dense = tf.keras.layers.Dense(units=units)
            else:
                raise ValueError('Invalid task type: {}'.format(type))
            self.tasks.append({'dense': dense, 'weight': task['weight'], 'type': type})
